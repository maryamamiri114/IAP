"""Step 1: precompute the intent-free answers y_bar ~ pi_ref(. | x_i) and their scores mu(y_bar).

pi_ref is the frozen initial checkpoint (the same model that IAP starts from), so
y_bar does not change during training and can be computed once.

Requires a running judge server (see iap/judge.py).

Example:
    python scripts/prepare_intent_free.py --model Qwen/Qwen2.5-7B-Instruct --chat \
        --output_dir data/train_qwen2.5-7b-it
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iap.data import DOMAINS, load_domains  # noqa: E402
from iap.generation import generate, load_llm  # noqa: E402
from iap.judge import RubricJudge  # noqa: E402
from iap.prompts import build_prompt, format_question  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Initial checkpoint = frozen reference policy")
    p.add_argument("--chat", action="store_true", help="Apply the chat template (instruction-tuned models)")
    p.add_argument("--domains", nargs="+", default=list(DOMAINS))
    p.add_argument("--split", default="train")
    p.add_argument("--max_samples", type=int, default=None, help="Per domain; for debugging")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge_url", default="http://localhost:8000/v1")
    p.add_argument("--judge_model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    ds = load_domains(args.domains, args.split, args.cache_dir, args.max_samples)
    prompts = [build_prompt("intent_free", format_question(q, n)) for q, n in zip(ds["question"], ds["narrative"])]

    llm = load_llm(args.model, args.tensor_parallel_size, seed=args.seed)
    answers = [o[0].strip() for o in generate(llm, prompts, args.chat, args.temperature, args.top_p,
                                              args.max_tokens, seed=args.seed)]
    del llm

    judge = RubricJudge(base_url=args.judge_url, model=args.judge_model)
    scores = judge.score_batch(list(zip(ds["question"], ds["narrative"], answers, ds["rubric_aspects"])))

    ds = ds.add_column("intent_free_answer", answers).add_column("intent_free_score", scores)
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(f"Saved {len(ds)} examples to {args.output_dir}; mean mu(y_bar) = {sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
