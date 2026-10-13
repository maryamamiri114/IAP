#!/usr/bin/env bash
# Full pipeline for one model: intent-free answers -> IAP training -> generation -> evaluation.
#
# Usage: bash scripts/run_model.sh <hf_model_id> <short_name> [--chat]
#   e.g. bash scripts/run_model.sh Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-it --chat
#
# Prerequisites:
#   * bash scripts/setup.sh && python scripts/export_lamp_json.py
#   * training judge running:  vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000
#     (on GPUs separate from the training GPUs)
#   * EVAL_JUDGE set to the final evaluator (Gemma-4-31B in the paper)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=$1
NAME=$2
CHAT=${3:-}

TRAIN_DATA=data/train_${NAME}
CKPT=checkpoints/iap-${NAME}
OUT=outputs/${NAME}

# 1. Intent-free answers from the frozen reference policy
python scripts/prepare_intent_free.py --model "$MODEL" $CHAT --output_dir "$TRAIN_DATA"

# 2. IAP training
accelerate launch --config_file configs/fsdp.yaml train.py \
  --model "$MODEL" $CHAT --train_data "$TRAIN_DATA" --output_dir "$CKPT"

# 3. Generation: IAP, prompting baselines, intent ablation
python generate.py --mode iap --model "$CKPT/final" $CHAT --output_dir "$OUT/iap"
for MODE in nonpers swi implicit no_intent; do
  python generate.py --mode "$MODE" --model "$MODEL" $CHAT --output_dir "$OUT/$MODE"
done
for MODE in explicit ingroup crossgroup; do
  python generate.py --mode "$MODE" --model "$MODEL" $CHAT --intents_dir "$OUT/iap" --output_dir "$OUT/$MODE"
done

# 4. Evaluation with the final judge
for MODE in iap nonpers swi implicit no_intent explicit ingroup crossgroup; do
  bash scripts/evaluate.sh "$OUT/$MODE"
done

# 5. Summary
python scripts/aggregate.py "$OUT"/{nonpers,swi,implicit,explicit,iap}
python scripts/aggregate.py "$OUT"/{no_intent,ingroup,crossgroup,explicit}
python scripts/aggregate.py "$OUT/iap" --intent_buckets
