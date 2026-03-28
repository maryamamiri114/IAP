"""
data_prep.py
------------
Downloads the LaMP-QA dataset from HuggingFace and prepares it for IAP training.

Dataset: alireza7/LaMP-QA
Each split (train / validation / test) has the following fields:
  - id          : unique question identifier
  - question    : the user's question (short form)
  - details     : the user's detailed narrative / context
  - profile     : list of user profile items (personal context)
  - aspects     : list of rubric aspect dicts used for evaluation
                  Each aspect has keys: aspect, reason, evidence

Output structure written to disk:
  data/
    train.jsonl
    validation.jsonl
    test.jsonl

Each JSONL line has:
  {
    "id": str,
    "question": str,          # short question text
    "details": str,           # narrative / user context
    "profile": [...],         # raw profile list
    "aspects": [...],         # rubric aspects for evaluation
    "input": str              # full formatted prompt passed to the model
  }
"""

import argparse
import json
import os

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Prompt template (matches the IAP paper's rollout prompt exactly)
# ---------------------------------------------------------------------------
IAP_SYSTEM_PROMPT = (
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


def format_profile(profile_items: list, num_contexts: int = 10) -> str:
    """Serialise up to `num_contexts` user profile items as a readable string."""
    items = profile_items[:num_contexts]
    if not items:
        return ""
    lines = ["User profile context:"]
    for idx, item in enumerate(items, 1):
        if isinstance(item, dict):
            text = item.get("text", item.get("content", str(item)))
        else:
            text = str(item)
        lines.append(f"  [{idx}] {text}")
    return "\n".join(lines)


def build_input_prompt(question: str, details: str, profile: list,
                       num_contexts: int = 10) -> str:
    """
    Builds the full string that is fed to the policy as the 'question'
    argument in the IAP prompt template.
    """
    profile_str = format_profile(profile, num_contexts)
    parts = []
    if profile_str:
        parts.append(profile_str)
    parts.append(f"Question: {question}")
    if details and details.strip():
        parts.append(f"Details: {details}")
    return "\n\n".join(parts)


def prepare_split(dataset_split, split_name: str, output_dir: str,
                  num_contexts: int = 10) -> None:
    """Process one dataset split and write a JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}.jsonl")

    written = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in dataset_split:
            question = row.get("question", "")
            details  = row.get("details", "")
            profile  = row.get("profile", [])
            aspects  = row.get("aspects", [])
            qid      = str(row.get("id", written))

            input_prompt = build_input_prompt(question, details, profile,
                                              num_contexts)

            record = {
                "id":       qid,
                "question": question,
                "details":  details,
                "profile":  profile,
                "aspects":  aspects,
                "input":    input_prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[{split_name}] Wrote {written} examples -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the LaMP-QA dataset for IAP training."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data",
        help="Directory where processed JSONL files will be saved."
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./cache",
        help="HuggingFace cache directory."
    )
    parser.add_argument(
        "--num_contexts", type=int, default=10,
        help="Number of user profile items to include in the prompt."
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation", "test"],
        help="Which dataset splits to prepare."
    )
    args = parser.parse_args()

    print("Loading LaMP-QA dataset from HuggingFace (alireza7/LaMP-QA) …")
    dataset = load_dataset("alireza7/LaMP-QA", cache_dir=args.cache_dir)

    for split in args.splits:
        if split not in dataset:
            print(f"WARNING: split '{split}' not found in dataset, skipping.")
            continue
        prepare_split(dataset[split], split, args.output_dir, args.num_contexts)

    print("\nData preparation complete.")
    print(f"Files saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
