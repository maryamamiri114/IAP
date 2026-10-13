#!/usr/bin/env bash
# Step 4: score a run with the official LaMP-QA evaluator.
#
# Usage: bash scripts/evaluate.sh <run_dir> [evaluator_llm] [split]
#   run_dir        folder written by generate.py (contains art.json, lifestyle.json, society.json)
#   evaluator_llm  final judge; the paper uses Gemma-4-31B (set EVAL_JUDGE or pass a path/HF id)
#   split          test (default) or validation
#
# Writes <run_dir>/<domain>_scores.json.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR=$1
EVALUATOR=${2:-${EVAL_JUDGE:?"Set EVAL_JUDGE or pass the evaluator model as the 2nd argument"}}
SPLIT=${3:-test}
TP=${TP:-1}

for DOMAIN in art lifestyle society; do
  python third_party/LaMP-QA/evaluate_responses.py \
    --evaluator_llm "$EVALUATOR" \
    --inputs_addr "data/lamp_qa/${DOMAIN}/${SPLIT}.json" \
    --response_addr "${RUN_DIR}/${DOMAIN}.json" \
    --score_addr "${RUN_DIR}/${DOMAIN}_scores.json" \
    --tensor_parallel_size "$TP"
done
