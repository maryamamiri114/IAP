"""Step 5: per-domain and macro scores, significance tests and the number-of-intents analysis.

Examples
  # scores of one or more runs
  python scripts/aggregate.py outputs/qwen2.5-7b-it/iap outputs/qwen2.5-7b-it/explicit

  # two-sided paired t-test of run A vs run B on per-question scores
  python scripts/aggregate.py outputs/qwen2.5-7b-it/iap --compare outputs/qwen2.5-7b-it/sft

  # IAP score by number of inferred intents (Low 1-2, Medium 3-4, High 5+)
  python scripts/aggregate.py outputs/qwen2.5-7b-it/iap --intent_buckets
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iap.parsing import count_intents  # noqa: E402

DOMAINS = ["art", "lifestyle", "society"]


def per_question(run_dir: str, domain: str) -> dict[str, float]:
    """id -> score. The official evaluator stores per-question ids after scoring."""
    with open(os.path.join(run_dir, f"{domain}_scores.json")) as f:
        scores = json.load(f)
    return {str(s["id"]): s["score"] for s in scores["per_question_scores"]}


def summarize(run_dir: str) -> dict[str, float]:
    out = {}
    for d in DOMAINS:
        q = per_question(run_dir, d)
        out[d] = sum(q.values()) / len(q)
    out["macro"] = sum(out[d] for d in DOMAINS) / len(DOMAINS)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--compare", default=None, help="Baseline run for paired t-tests against the first run")
    p.add_argument("--intent_buckets", action="store_true")
    args = p.parse_args()

    print(f"{'run':<50} " + " ".join(f"{c:>9}" for c in DOMAINS + ["macro"]))
    for run in args.runs:
        s = summarize(run)
        print(f"{run:<50} " + " ".join(f"{s[c]:>9.4f}" for c in DOMAINS + ["macro"]))

    if args.compare:
        from scipy.stats import ttest_rel

        a_run, b_run = args.runs[0], args.compare
        print(f"\nPaired t-test: {a_run} vs {b_run}")
        for d in DOMAINS:
            a, b = per_question(a_run, d), per_question(b_run, d)
            ids = sorted(set(a) & set(b))
            res = ttest_rel([a[i] for i in ids], [b[i] for i in ids])
            ma, mb = sum(a[i] for i in ids) / len(ids), sum(b[i] for i in ids) / len(ids)
            delta = 100 * (ma - mb) / mb if mb else float("nan")
            print(f"  {d:<10} n={len(ids):<5} A={ma:.4f} B={mb:.4f} rel.gain={delta:+.2f}%  p={res.pvalue:.2e}")
        print("  (No multiple-comparison correction is applied.)")

    if args.intent_buckets:
        run = args.runs[0]
        buckets = {"Low (1-2)": [], "Medium (3-4)": [], "High (5+)": []}
        for d in DOMAINS:
            scores = per_question(run, d)
            with open(os.path.join(run, f"{d}.json")) as f:
                outputs = json.load(f)
            for qid, resp in outputs.items():
                n = count_intents(resp[0].get("intent"))
                if n == 0 or qid not in scores:
                    continue
                key = "Low (1-2)" if n <= 2 else "Medium (3-4)" if n <= 4 else "High (5+)"
                buckets[key].append(scores[qid])
        print(f"\nScore by number of inferred intents ({run})")
        for k, v in buckets.items():
            mean = sum(v) / len(v) if v else float("nan")
            print(f"  {k:<13} n={len(v):<5} mean={mean:.4f}")


if __name__ == "__main__":
    main()
