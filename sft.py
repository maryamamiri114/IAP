"""SFT baseline (rejection-sampling / reward-ranked fine-tuning):
for each training question, sample K responses, keep the one with the highest
training-judge score, and fine-tune on these silver labels.

Stage 1 (build silver labels; needs the judge server):
    python sft.py build --model Qwen/Qwen2.5-7B-Instruct --chat --output data/sft_qwen2.5-7b-it

Stage 2 (fine-tune):
    accelerate launch --config_file configs/fsdp.yaml sft.py train \
        --model Qwen/Qwen2.5-7B-Instruct --chat --data data/sft_qwen2.5-7b-it \
        --output_dir checkpoints/sft-qwen2.5-7b-it

At test time, generate with the same --prompt_mode used here.
"""

import argparse

from iap.data import DOMAINS, load_domains
from iap.prompts import build_prompt, format_question


def build(args):
    from iap.generation import generate, load_llm
    from iap.judge import RubricJudge

    ds = load_domains(args.domains, "train", args.cache_dir, args.max_samples)
    prompts = [build_prompt(args.prompt_mode, format_question(q, n)) for q, n in zip(ds["question"], ds["narrative"])]

    llm = load_llm(args.model, args.tensor_parallel_size, seed=args.seed)
    samples = generate(llm, prompts, args.chat, args.temperature, args.top_p, args.max_tokens,
                       n=args.num_samples, seed=args.seed)
    del llm

    judge = RubricJudge(base_url=args.judge_url, model=args.judge_model)
    flat = [(q, n, s, a) for q, n, a, ss in zip(ds["question"], ds["narrative"], ds["rubric_aspects"], samples)
            for s in ss]
    flat_scores = judge.score_batch(flat)

    best, best_scores, k = [], [], args.num_samples
    for i, ss in enumerate(samples):
        sc = flat_scores[i * k:(i + 1) * k]
        j = max(range(k), key=lambda t: sc[t])
        best.append(ss[j])
        best_scores.append(sc[j])

    out = ds.add_column("prompt_text", prompts).add_column("completion_text", best) \
            .add_column("silver_score", best_scores)
    out.save_to_disk(args.output)
    print(f"Saved {len(out)} silver labels; mean best score = {sum(best_scores) / len(best_scores):.4f}")


def train(args):
    from datasets import load_from_disk
    from trl import SFTConfig, SFTTrainer

    ds = load_from_disk(args.data)

    def to_pc(ex):
        if args.chat:
            return {"prompt": [{"role": "user", "content": ex["prompt_text"]}],
                    "completion": [{"role": "assistant", "content": ex["completion_text"]}]}
        return {"prompt": ex["prompt_text"], "completion": ex["completion_text"]}

    ds = ds.map(to_pc, remove_columns=ds.column_names)
    config = SFTConfig(
        output_dir=args.output_dir, num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True, bf16=True, logging_steps=10, save_strategy="epoch",
        completion_only_loss=True, seed=args.seed, report_to="none",
    )
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=ds)
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)

    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--chat", action="store_true")
    b.add_argument("--output", required=True)
    b.add_argument("--prompt_mode", default="implicit", help="Prompt used to sample responses")
    b.add_argument("--domains", nargs="+", default=list(DOMAINS))
    b.add_argument("--num_samples", type=int, default=5)
    b.add_argument("--temperature", type=float, default=1.0)
    b.add_argument("--top_p", type=float, default=1.0)
    b.add_argument("--max_tokens", type=int, default=2048)
    b.add_argument("--tensor_parallel_size", type=int, default=1)
    b.add_argument("--judge_url", default="http://localhost:8000/v1")
    b.add_argument("--judge_model", default="Qwen/Qwen2.5-32B-Instruct")
    b.add_argument("--max_samples", type=int, default=None)
    b.add_argument("--cache_dir", default=None)
    b.add_argument("--seed", type=int, default=42)

    t = sub.add_parser("train")
    t.add_argument("--model", required=True)
    t.add_argument("--chat", action="store_true")
    t.add_argument("--data", required=True)
    t.add_argument("--output_dir", required=True)
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--learning_rate", type=float, default=1e-5)
    t.add_argument("--warmup_ratio", type=float, default=0.03)
    t.add_argument("--per_device_train_batch_size", type=int, default=4)
    t.add_argument("--gradient_accumulation_steps", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    build(args) if args.stage == "build" else train(args)


if __name__ == "__main__":
    main()
