"""Step 3: generate test-set answers for IAP, the baselines and the intent ablation.

Modes
  iap         trained IAP checkpoint, tagged prompt; the <answer> tag is the output
  nonpers     Non-personalized baseline (question only, no narrative)
  swi         SWI baseline
  implicit    Implicit Intent Personalization
  explicit    Explicit Intent Personalization (IAP intents, untrained model) = "Selected" in Table 2
  no_intent   Intent ablation: personalized prompt, no intent
  ingroup     Intent ablation: random IAP intent from another question in the same domain
  crossgroup  Intent ablation: random IAP intent from a question in a different domain

explicit / ingroup / crossgroup need --intents_dir: the output folder of an `iap` run.

Output: <output_dir>/<domain>.json in the official LaMP-QA response format,
{id: [{"output": answer, ...}]}, so it can be scored by evaluate_responses.py.

Example:
    python generate.py --mode iap --model checkpoints/iap-qwen2.5-7b-it/final --chat \
        --output_dir outputs/qwen2.5-7b-it/iap
"""

import argparse
import json
import os
import random

from iap.data import DOMAINS, load_domain
from iap.generation import generate, load_llm
from iap.parsing import parse_rollout
from iap.prompts import build_prompt, format_question

MODES = ["iap", "nonpers", "swi", "implicit", "explicit", "no_intent", "ingroup", "crossgroup"]
NEEDS_INTENTS = {"explicit", "ingroup", "crossgroup"}


def load_intents(intents_dir: str, domains: list[str]) -> dict[str, dict[str, str]]:
    intents = {}
    for d in domains:
        with open(os.path.join(intents_dir, f"{d}.json")) as f:
            intents[d] = {qid: r[0]["intent"] for qid, r in json.load(f).items() if r[0].get("intent")}
    return intents


def pick_intent(mode, domain, qid, intents, rng):
    if mode == "explicit":
        return intents[domain].get(qid)
    if mode == "ingroup":
        pool = [i for k, i in intents[domain].items() if k != qid]
    else:  # crossgroup
        pool = [i for d, m in intents.items() if d != domain for i in m.values()]
    return rng.choice(pool) if pool else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--chat", action="store_true")
    p.add_argument("--domains", nargs="+", default=list(DOMAINS))
    p.add_argument("--split", default="test")
    p.add_argument("--intents_dir", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    if args.mode in NEEDS_INTENTS and not args.intents_dir:
        p.error(f"--mode {args.mode} needs --intents_dir (output of an `iap` run)")
    # cross-group sampling needs every domain's intents, not only the generated ones
    intents = load_intents(args.intents_dir, list(DOMAINS)) if args.mode in NEEDS_INTENTS else None
    rng = random.Random(args.seed)

    llm = load_llm(args.model, args.tensor_parallel_size, seed=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for domain in args.domains:
        ds = load_domain(domain, args.split, args.cache_dir, args.max_samples)
        prompts, used_intents = [], []
        for ex in ds:
            # The Non-personalized baseline sees no user information (question only).
            narrative = None if args.mode == "nonpers" else ex["narrative"]
            x = format_question(ex["question"], narrative)
            intent = pick_intent(args.mode, domain, ex["id"], intents, rng) if intents else None
            if args.mode in NEEDS_INTENTS and intent is None:
                intent = "None"  # IAP produced no intent for this question
            prompts.append(build_prompt(args.mode, x, intent))
            used_intents.append(intent)

        raw = [o[0] for o in generate(llm, prompts, args.chat, args.temperature, args.top_p,
                                      args.max_tokens, seed=args.seed)]

        results = {}
        for ex, text, intent in zip(ds, raw, used_intents):
            if args.mode == "iap":
                parsed = parse_rollout(text)
                results[ex["id"]] = [{"output": parsed["answer"] or "", "intent": parsed["intent"],
                                      "well_formed": parsed["well_formed"], "raw": text}]
            else:
                results[ex["id"]] = [{"output": text.strip(), "intent": intent}]

        with open(os.path.join(args.output_dir, f"{domain}.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{domain}] wrote {len(results)} responses")


if __name__ == "__main__":
    main()
