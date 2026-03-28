"""
training.py
-----------
Implements the IAP (Intent-Aware Personalisation) training loop from the paper.

Pipeline per training step
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Sample a mini-batch of (question, aspects) pairs from the training JSONL.
2. For each instance, sample G rollouts from the current policy.
   Each rollout follows the structured format:
       <think>…</think>
       <intent>…</intent>
       <think>…</think>
       <answer>…</answer>
3. Compute the three reward components:
     r_pers      – LaMP-QA rubric score   (Qwen2.5-32B judge via vLLM)
     r_contrast  – ROUGE-L similarity to intent-free reference answer (penalty)
     r_length    – intent length penalty
4. Compute group-relative advantages (DAPO style).
5. Update the policy with the DAPO clipped surrogate objective.
6. Periodically save a checkpoint and run optional validation.

Usage
-----
python training.py \
    --policy_model   Qwen/Qwen2.5-7B-Instruct \
    --evaluator_model Qwen/Qwen2.5-32B-Instruct \
    --train_path     data/train.jsonl \
    --val_path       data/validation.jsonl \
    --output_dir     checkpoints/iap \
    [--num_rollouts 8] \
    [--batch_size 4] \
    [--max_steps 1000] \
    [--lambda1 1.0] [--lambda2 0.1] [--lambda3 0.01] \
    [--intent_length_threshold 200] \
    [--mu_updates 1] \
    [--lr 1e-6] \
    [--kl_coef 0.01] \
    [--clip_eps 0.2] \
    [--save_steps 100] \
    [--eval_steps 100] \
    [--tensor_parallel_size 1]
"""

import argparse
import json
import os
import re
import math
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Inline evaluator (same as evaluation.py)
import json5
from vllm import SamplingParams as vSamplingParams

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LaMP-QA evaluator (inline, identical to evaluation.py)
# ---------------------------------------------------------------------------

_EVAL_PROMPT_SYSTEM = """You are a fair and insightful judge with exceptional reasoning and analytical abilities. Your task is to evaluate a user's question, a generated response to that question, and an aspect that is important to the user. Based on this information, identify if the aspect is addressed in the generated response. Provide a clear and accurate assessment.

# your input:
- question: the question asked by the user.
- details: the detailed explanation of the question from the user.
- response: a generated response to the user's question
- aspect: the aspect that is important to the user, consisting of the following fields:
  - aspect: the title for the aspect.
  - reason: the reason that this aspect is important for the user.
  - evidence: the evidence from the user detailed explanation that the aspect extracted from.

# your output: Your output should be only a valid json object in ```json ``` block without any explanations that contains the following fields:
- match_score: A score between 0 to 2 that indicates how well the generated response addresses this aspect, where: 0 means the response does not cover this aspect, 1 means the response somewhat covers this aspect, and 2 means the response covers this aspect very well.
"""

_EVAL_PROMPT_USER = """
question: {question}

details: {details}

response: {response}

aspect: {aspects}

Your output should be only a valid json object in ```json ``` block without any explanations.
"""


def _parse_json_eval(json_str: str):
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(json_str, strict=False)
    except Exception:
        pass
    try:
        return json5.loads(json_str)
    except Exception:
        raise ValueError(f"Could not parse: {json_str[:200]}")


def _build_eval_prompts(queries, responses, details_list, aspects_list, tokenizer):
    prompts, ids = [], []
    for i, (q, r, d, asps) in enumerate(zip(queries, responses, details_list, aspects_list)):
        for j, asp in enumerate(asps):
            asp_str = (
                f'-aspect: {asp["aspect"]}\n'
                f' -reason: {asp["reason"]}\n'
                f' -evidence: {asp["evidence"]}'
            )
            conv = [
                {"role": "system", "content": _EVAL_PROMPT_SYSTEM},
                {"role": "user", "content": _EVAL_PROMPT_USER.format(
                    question=q, details=d, response=r, aspects=asp_str)},
            ]
            prompts.append(
                tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            )
            ids.append({"q_id": i, "a_id": j})
    return ids, prompts


def compute_personalized_rewards(queries, rollout_answers, details_list,
                                  aspects_list, eval_llm, max_retries=20):
    """
    Compute µ(x, ŷ, E) for a batch of rollout answers.
    Returns a list of float scores in [0, 1], one per (query, rollout_answer) pair.
    """
    tokenizer = eval_llm.get_tokenizer()
    temperature = 0.0
    retries = 0
    ids, prompts = _build_eval_prompts(
        queries, rollout_answers, details_list, aspects_list, tokenizer
    )
    outputs_dict: dict = {}

    while prompts:
        retries += 1
        sp = vSamplingParams(temperature=temperature, top_p=0.95, max_tokens=512)
        outputs = eval_llm.generate(prompts, sp)
        wrongs = []
        for id_, prompt, out in zip(ids, prompts, outputs):
            q_id, a_id = id_["q_id"], id_["a_id"]
            if q_id not in outputs_dict:
                outputs_dict[q_id] = {}
            try:
                obj = _parse_json_eval(out.outputs[0].text)
                outputs_dict[q_id][a_id] = float(obj["match_score"])
            except Exception:
                if retries > max_retries:
                    outputs_dict[q_id][a_id] = 0.0
                    continue
                wrongs.append((id_, prompt))
        prompts = [p for _, p in wrongs]
        ids     = [i for i, _ in wrongs]
        if temperature < 1.0:
            temperature = min(temperature + 0.1, 1.0)

    scores = []
    for i, aspects in enumerate(aspects_list):
        raw = sum(outputs_dict[i].get(j, 0.0) for j in range(len(aspects)))
        scores.append(raw / (len(aspects) * 2) if aspects else 0.0)
    return scores


# ---------------------------------------------------------------------------
# Rollout parsing helpers
# ---------------------------------------------------------------------------

def extract_tag(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag>; returns '' if not found."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_answer(rollout: str) -> str:
    return extract_tag(rollout, "answer")


def extract_intent(rollout: str) -> str:
    return extract_tag(rollout, "intent")


# ---------------------------------------------------------------------------
# ROUGE-L helper
# ---------------------------------------------------------------------------

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    return _rouge.score(ref, pred)["rougeL"].fmeasure


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_rewards(
    queries: List[str],
    rollout_batches: List[List[str]],   # shape: [batch_size, G]
    ref_answers: List[str],
    details_list: List[str],
    aspects_list: List[List[Dict]],
    eval_llm,
    lambda1: float = 1.0,
    lambda2: float = 0.1,
    lambda3: float = 0.01,
    intent_length_threshold: int = 200,
    max_eval_retries: int = 20,
) -> List[List[float]]:
    """
    Compute combined IAP reward for every rollout in the batch.

    Returns rewards[i][g] for instance i, rollout g.
    """
    batch_size = len(queries)
    G = len(rollout_batches[0])

    # Flatten rollouts for batch evaluation
    flat_queries   = [q   for q   in queries  for _ in range(G)]
    flat_details   = [d   for d   in details_list  for _ in range(G)]
    flat_aspects   = [asp for asp in aspects_list   for _ in range(G)]
    flat_answers   = [extract_answer(rollout_batches[i][g])
                      for i in range(batch_size) for g in range(G)]

    # --- r_pers (personalized rubric reward) --------------------------------
    r_pers_flat = compute_personalized_rewards(
        flat_queries, flat_answers, flat_details, flat_aspects,
        eval_llm, max_retries=max_eval_retries,
    )

    # --- r_contrast (ROUGE-L penalty vs intent-free reference) --------------
    r_contrast_flat = [
        rouge_l(flat_answers[k], ref_answers[k // G])
        for k in range(batch_size * G)
    ]

    # --- r_length (intent length penalty) -----------------------------------
    r_length_flat = []
    for i in range(batch_size):
        for g in range(G):
            intent_text = extract_intent(rollout_batches[i][g])
            excess = max(0, len(intent_text.split()) - intent_length_threshold)
            r_length_flat.append(-excess)

    # --- Combine ------------------------------------------------------------
    rewards = []
    for i in range(batch_size):
        row = []
        for g in range(G):
            k = i * G + g
            r = (
                lambda1 * r_pers_flat[k]
                - lambda2 * r_contrast_flat[k]
                + lambda3 * r_length_flat[k]
            )
            row.append(r)
        rewards.append(row)

    return rewards


# ---------------------------------------------------------------------------
# Group-relative advantage (DAPO style)
# ---------------------------------------------------------------------------

def compute_advantages(rewards: List[List[float]]) -> List[List[float]]:
    """
    Normalise rewards within each group to zero mean / unit std (DAPO).
    Groups where all rewards are identical get zero advantage.
    """
    advantages = []
    for group in rewards:
        t = torch.tensor(group, dtype=torch.float32)
        mean = t.mean()
        std  = t.std(unbiased=False)
        if std < 1e-8:
            advantages.append([0.0] * len(group))
        else:
            adv = ((t - mean) / (std + 1e-8)).tolist()
            advantages.append(adv)
    return advantages


# ---------------------------------------------------------------------------
# DAPO policy loss
# ---------------------------------------------------------------------------

def compute_dapo_loss(
    policy_model,
    ref_model,
    tokenizer,
    rollouts: List[str],
    advantages: List[float],
    kl_coef: float = 0.01,
    clip_eps: float = 0.2,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute the DAPO clipped surrogate loss for a flat list of rollouts.

    For each rollout we compute:
      ratio = exp(log π_θ(y|x) - log π_old(y|x))
      clipped_loss = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    with a KL penalty against the frozen reference model.

    NOTE: Here π_old ≈ π_θ at the start of each μ_upd step (on-policy),
    so log π_old is computed from the reference snapshot passed in.
    For a full off-policy multi-step update you would cache the old log-probs
    from the rollout phase; this implementation handles the μ_upd=1 default.
    """
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for rollout_text, adv in zip(rollouts, advantages):
        if not rollout_text.strip():
            continue

        enc = tokenizer(
            rollout_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        input_ids = enc["input_ids"]

        with torch.no_grad():
            ref_out  = ref_model(**enc)
            ref_logp = F.log_softmax(ref_out.logits[:, :-1, :], dim=-1)
            ref_seq_logp = ref_logp.gather(
                2, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1)

        pol_out  = policy_model(**enc)
        pol_logp = F.log_softmax(pol_out.logits[:, :-1, :], dim=-1)
        pol_seq_logp = pol_logp.gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1).sum(dim=-1)

        # DAPO clipped surrogate
        ratio = torch.exp(pol_seq_logp - ref_seq_logp.detach())
        adv_t = torch.tensor(adv, device=device)
        loss_unclipped = -ratio * adv_t
        loss_clipped   = -torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
        policy_loss    = torch.max(loss_unclipped, loss_clipped).mean()

        # KL penalty: KL(π_θ || π_ref)  ≈  π_θ * (log π_θ - log π_ref)
        kl = (pol_seq_logp - ref_seq_logp.detach()).mean()

        total_loss += policy_loss + kl_coef * kl
        count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_iap_prompt(input_text: str) -> str:
    """Wrap the pre-formatted input with the IAP system instruction."""
    system = (
        "Your task is to generate a personalized response to the user's question. "
        "To do this, you can perform a series of actions, including thinking in "
        "<think> and </think> tags, generating potential user intent(s) by producing "
        "a non-empty list in <intent> and </intent> tags, and finally providing the "
        "answer in <answer> and </answer> tags.\n"
        "You need to first think about the question and how to generate a personalized "
        "answer for the user based on the user intent and needs.\n"
        "The thinking process should be inside <think> and </think> tags.\n"
        "Then you generate the potential intent(s) and put them in <intent> tags.\n"
        "Next, think again using those intent(s) to guide answer planning and generation, "
        "again inside <think> and </think> tags.\n"
        "Nothing should be outside the mentioned tags except the initial question.\n"
    )
    return f"{system}\nNow, answer the following question: {input_text}"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IAP training with DAPO + LaMP-QA rewards.")
    parser.add_argument("--policy_model",        type=str, required=True)
    parser.add_argument("--evaluator_model",     type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--train_path",          type=str, required=True)
    parser.add_argument("--val_path",            type=str, default=None)
    parser.add_argument("--output_dir",          type=str, default="checkpoints/iap")
    parser.add_argument("--num_rollouts",        type=int, default=8,   help="G in the paper")
    parser.add_argument("--batch_size",          type=int, default=4)
    parser.add_argument("--max_steps",           type=int, default=1000)
    parser.add_argument("--lambda1",             type=float, default=1.0)
    parser.add_argument("--lambda2",             type=float, default=0.1)
    parser.add_argument("--lambda3",             type=float, default=0.01)
    parser.add_argument("--intent_length_threshold", type=int, default=200)
    parser.add_argument("--mu_updates",          type=int, default=1,   help="μ_upd inner steps")
    parser.add_argument("--lr",                  type=float, default=1e-6)
    parser.add_argument("--kl_coef",             type=float, default=0.01)
    parser.add_argument("--clip_eps",            type=float, default=0.2)
    parser.add_argument("--save_steps",          type=int, default=100)
    parser.add_argument("--eval_steps",          type=int, default=100)
    parser.add_argument("--max_new_tokens",      type=int, default=1024)
    parser.add_argument("--tensor_parallel_size",type=int, default=1)
    parser.add_argument("--seed",                type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load data ----------------------------------------------------------
    logger.info("Loading training data …")
    train_data = load_jsonl(args.train_path)
    val_data   = load_jsonl(args.val_path) if args.val_path else []

    # --- Load policy (HF) and frozen reference model -----------------------
    logger.info(f"Loading policy model: {args.policy_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    policy_model.train()

    logger.info("Loading frozen reference model …")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # --- Load evaluator LLM (vLLM) -----------------------------------------
    logger.info(f"Loading evaluator model: {args.evaluator_model}")
    eval_llm = LLM(
        model=args.evaluator_model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # --- vLLM rollout sampler (uses policy weights via HF; we re-use vLLM
    #     for fast generation and sync weights each step) --------------------
    # For simplicity we use HuggingFace generate for rollouts so we don't
    # need to maintain a separate vLLM process for the policy.
    # For large-scale runs, replace this with vLLM + weight sync.

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)

    # --- Training loop ------------------------------------------------------
    logger.info("Starting IAP training …")
    step = 0
    data_iter = iter(random.sample(train_data, len(train_data)))

    while step < args.max_steps:
        # Sample mini-batch
        batch = []
        for _ in range(args.batch_size):
            try:
                batch.append(next(data_iter))
            except StopIteration:
                data_iter = iter(random.sample(train_data, len(train_data)))
                batch.append(next(data_iter))

        queries      = [r["question"]           for r in batch]
        details_list = [r.get("details", "")    for r in batch]
        aspects_list = [r["aspects"]            for r in batch]
        inputs       = [build_iap_prompt(r["input"]) for r in batch]

        # --- Sample G rollouts per instance --------------------------------
        rollout_batches: List[List[str]] = []   # [batch_size, G]
        for inp in inputs:
            enc = tokenizer(inp, return_tensors="pt", truncation=True,
                            max_length=1024).to(device)
            group = []
            with torch.no_grad():
                for _ in range(args.num_rollouts):
                    out = policy_model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    generated = tokenizer.decode(
                        out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    group.append(generated)
            rollout_batches.append(group)

        # --- Generate intent-free reference answers (frozen ref model) -----
        ref_answers = []
        for inp in inputs:
            bare_inp = inp.split("Now, answer the following question:")[-1].strip()
            enc = tokenizer(bare_inp, return_tensors="pt", truncation=True,
                            max_length=1024).to(device)
            with torch.no_grad():
                out = ref_model.generate(
                    **enc,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            ref_answers.append(
                tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            )

        # --- Compute rewards -----------------------------------------------
        rewards = compute_rewards(
            queries=queries,
            rollout_batches=rollout_batches,
            ref_answers=ref_answers,
            details_list=details_list,
            aspects_list=aspects_list,
            eval_llm=eval_llm,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            intent_length_threshold=args.intent_length_threshold,
        )

        # --- Group-relative advantages -------------------------------------
        advantages = compute_advantages(rewards)

        # --- DAPO policy update (μ_upd inner steps) ------------------------
        for _ in range(args.mu_updates):
            optimizer.zero_grad()

            # Flatten rollouts and their advantages
            flat_rollouts = [
                rollout_batches[i][g]
                for i in range(len(batch))
                for g in range(args.num_rollouts)
            ]
            flat_advantages = [
                advantages[i][g]
                for i in range(len(batch))
                for g in range(args.num_rollouts)
            ]

            loss = compute_dapo_loss(
                policy_model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                rollouts=flat_rollouts,
                advantages=flat_advantages,
                kl_coef=args.kl_coef,
                clip_eps=args.clip_eps,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

        step += 1
        mean_reward = sum(sum(r) / len(r) for r in rewards) / len(rewards)
        logger.info(f"Step {step}/{args.max_steps} | loss={loss.item():.4f} | mean_reward={mean_reward:.4f}")

        # --- Save checkpoint -----------------------------------------------
        if step % args.save_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Checkpoint saved to {ckpt_dir}")

        # --- Validation ----------------------------------------------------
        if args.val_path and step % args.eval_steps == 0 and val_data:
            logger.info("Running validation …")
            val_responses = {}
            policy_model.eval()
            with torch.no_grad():
                for rec in val_data[:50]:   # quick validation subset
                    inp = build_iap_prompt(rec["input"])
                    enc = tokenizer(inp, return_tensors="pt", truncation=True,
                                    max_length=1024).to(device)
                    out = policy_model.generate(
                        **enc,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    generated = tokenizer.decode(
                        out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
                    )
                    answer = extract_answer(generated) or generated
                    val_responses[rec["id"]] = [{"output": answer}]
            policy_model.train()

            # Write temp response file and evaluate
            tmp_resp = os.path.join(args.output_dir, f"val_resp_step{step}.json")
            with open(tmp_resp, "w") as f:
                json.dump(val_responses, f)

            val_queries      = [r["question"]        for r in val_data[:50]]
            val_details      = [r.get("details", "") for r in val_data[:50]]
            val_aspects      = [r["aspects"]         for r in val_data[:50]]
            val_answers_list = [val_responses[r["id"]][0]["output"] for r in val_data[:50]]

            val_scores = compute_personalized_rewards(
                val_queries, val_answers_list, val_details, val_aspects, eval_llm
            )
            logger.info(f"Validation mean score @ step {step}: {sum(val_scores)/len(val_scores):.4f}")

    # --- Final save ---------------------------------------------------------
    final_dir = os.path.join(args.output_dir, "final")
    policy_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
