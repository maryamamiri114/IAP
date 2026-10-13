"""Exports LaMP-QA splits to the JSON files expected by the official evaluate_responses.py.

Writes data/lamp_qa/<domain>/<split>.json (profile column dropped).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iap.data import DOMAINS, load_domain  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--output_dir", default="data/lamp_qa")
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    for domain in DOMAINS:
        os.makedirs(os.path.join(args.output_dir, domain), exist_ok=True)
        for split in args.splits:
            ds = load_domain(domain, split, args.cache_dir)
            path = os.path.join(args.output_dir, domain, f"{split}.json")
            with open(path, "w") as f:
                json.dump(ds.to_list(), f)
            print(f"{path}: {len(ds)} examples")


if __name__ == "__main__":
    main()
