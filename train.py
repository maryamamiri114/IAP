"""Step 2: IAP training with TRL's GRPOTrainer using the DAPO token-level loss.

Defaults follow the paper's configuration (Section 4):
  G = 5 rollouts, 200 steps, lr 1e-5, warm-up ratio 0.285, beta = 0.001, epsilon = 0.2,
  lambda1 = 1.0, lambda2 = 0.5, lambda3 = 0.01, lambda (BERTScore) = 0.01, L = 400,
  2048-token completion budget, vLLM rollouts (TP 1, GPU memory 0.6, T = 1.0, top-p = 1.0),
  gradient checkpointing, FSDP (configs/fsdp.yaml).

Note on DAPO: TRL implements DAPO's token-level loss, decoupled clipping
(--epsilon_high) and overlong masking (--mask_truncated_completions), but NOT
dynamic sampling. With the paper's beta = 0.001 and epsilon_high = epsilon = 0.2,
this run uses a KL term and symmetric clipping; see README.

Example (8 GPUs):
    accelerate launch --config_file configs/fsdp.yaml train.py \
        --model Qwen/Qwen2.5-7B-Instruct --chat \
        --train_data data/train_qwen2.5-7b-it --output_dir checkpoints/iap-qwen2.5-7b-it
"""

import argparse
import os

from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from iap.judge import RubricJudge
from iap.prompts import build_prompt, format_question
from iap.rewards import IAPRewards


def parse_args():
    p = argparse.ArgumentParser()
    # model / data
    p.add_argument("--model", required=True)
    p.add_argument("--chat", action="store_true", help="Use the chat template (instruction-tuned models)")
    p.add_argument("--train_data", required=True, help="Output of scripts/prepare_intent_free.py")
    p.add_argument("--output_dir", required=True)
    # reward
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=0.5)
    p.add_argument("--lambda3", type=float, default=0.01)
    p.add_argument("--lambda_bertscore", type=float, default=0.01)
    p.add_argument("--intent_len_threshold", type=int, default=400, help="L, in tokens")
    p.add_argument("--bertscore_model", default="microsoft/deberta-v3-large")
    p.add_argument("--bertscore_layers", type=int, default=18)
    p.add_argument("--judge_url", default="http://localhost:8000/v1")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--judge_workers", type=int, default=64)
    # optimisation
    p.add_argument("--num_generations", type=int, default=5, help="G")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--per_device_train_batch_size", type=int, default=10)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.285)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epsilon_high", type=float, default=0.2, help="DAPO clip-higher uses 0.28")
    p.add_argument("--num_iterations", type=int, default=1, help="mu_upd: updates per batch")
    p.add_argument("--mask_truncated_completions", action="store_true")
    p.add_argument("--max_completion_length", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.6)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", default="none")
    return p.parse_args()


def main():
    args = parse_args()

    ds = load_from_disk(args.train_data)

    def to_prompt(ex):
        text = build_prompt("iap", format_question(ex["question"], ex["narrative"]))
        return {"prompt": [{"role": "user", "content": text}] if args.chat else text}

    ds = ds.map(to_prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    judge = RubricJudge(base_url=args.judge_url, model=args.judge_model, max_workers=args.judge_workers)
    rewards = IAPRewards(judge, tokenizer, lam=args.lambda_bertscore,
                         intent_len_threshold=args.intent_len_threshold,
                         bertscore_model=args.bertscore_model, bertscore_layers=args.bertscore_layers)

    config = GRPOConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        # optimisation
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        # GRPO / DAPO
        loss_type="dapo",
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        mask_truncated_completions=args.mask_truncated_completions,
        reward_weights=[args.lambda1, -args.lambda2, -args.lambda3],
        # rollouts
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        # logging
        logging_steps=1,
        log_completions=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to=args.report_to,
    )

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=[rewards.r_pers, rewards.r_contrast, rewards.r_length],
        args=config,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
