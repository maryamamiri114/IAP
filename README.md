## Title
**Training LLMs with Reinforcement Learning for Intent-Aware Personalized Question Answering**

## Abstract
Personalized question answering can be substantially advanced by aligning answers with the user's underlying intent, the implicit ``why'' that drives a query beyond what is explicitly asked. However, most existing intent modeling approaches rely on closed-domain taxonomies or surface-level cues, and are typically evaluated in settings that assume access to multi-turn conversational context or rich user profiles. As a result, these methods struggle in single-turn scenarios, where the user's latent goal must be inferred from minimal textual input alone. To address this gap, we propose IAP (**I**ntent **A**ware **P**ersonalization), a reinforcement learning framework that trains models to infer implicit user intent directly from a single-turn question and incorporate it into their reasoning process to generate personalized, intent-grounded answers. By optimizing intent-aware answer trajectories under a personalized reward function, IAP reinforces generation paths that make implicit user intent explicit and produce responses that better align with the user's underlying goal and personalized needs as captured by the reward model. Through extensive experiments on the LaMP-QA benchmark across six models, IAP consistently outperforms all baselines, achieving an average macro-score gain of approximately 7.5\% over the strongest competitor, demonstrating that modeling implicit user intent within the training objective is a promising direction for PQA.


## Repository structure

```
IAP/
├── iap/
│   ├── prompts.py        # IAP, baseline and ablation prompts
│   ├── parsing.py        # tag parsing, format checking, intent counting
│   ├── data.py           # LaMP-QA loading
│   ├── judge.py          # rubric LLM judge (official LaMP-QA prompt) via a vLLM server
│   ├── rewards.py        # r_pers, r_contrast, r_length
│   └── generation.py     # offline vLLM generation
├── scripts/
│   ├── setup.sh                  # install + clone the official LaMP-QA evaluator
│   ├── export_lamp_json.py       # dataset -> JSON files for the official evaluator
│   ├── prepare_intent_free.py    # step 1: intent-free answers and their scores
│   ├── evaluate.sh               # step 4: official LaMP-QA evaluation
│   ├── aggregate.py              # step 5: tables, paired t-tests, #intents analysis
│   └── run_model.sh              # full pipeline for one model
├── configs/fsdp.yaml     # accelerate FSDP config
├── train.py              # step 2: IAP training
├── generate.py           # step 3: IAP, baselines and intent ablation
└── sft.py                # SFT baseline (best-of-K silver labels)
```

## Setup

```bash
bash scripts/setup.sh                 # pip install + clone LaMP-QA into third_party/
python scripts/export_lamp_json.py    # writes data/lamp_qa/<domain>/{validation,test}.json
```

The training reward imports the judge prompt and JSON parser directly from the official LaMP-QA repository, so the reward uses exactly the benchmark's evaluation prompt.

Hardware used in the paper: NVIDIA H100 80GB GPUs.

## Data

[LaMP-QA](https://huggingface.co/datasets/alireza7/LaMP-QA) (Salemi & Zamani, 2025), three domains:

| Short name | LaMP-QA subset |
|---|---|
| `art` | Art_and_Entertainment |
| `lifestyle` | Lifestyle_and_Personal_Development |
| `society` | Society_and_Culture |

Each example has a `question`, the user's `narrative`, and `rubric_aspects` used for scoring. The model input $x_i$ is the question plus its narrative; the user `profile` is not used.

## Running

### 0. Start the training judge

The training reward uses Qwen2.5-32B-Instruct as the judge, served on GPUs separate from the training GPUs:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000 --tensor-parallel-size 2
```

### 1. Intent-free answers

The frozen reference model (the initial checkpoint) answers every training question without any intent step. The answers $\bar y$ and their scores $\mu(\bar y)$ are computed once and stored with the training data.

```bash
python scripts/prepare_intent_free.py --model Qwen/Qwen2.5-7B-Instruct --chat \
    --output_dir data/train_qwen2.5-7b-it
```

Use `--chat` for instruction-tuned models and omit it for base models.

### 2. IAP training

```bash
accelerate launch --config_file configs/fsdp.yaml train.py \
    --model Qwen/Qwen2.5-7B-Instruct --chat \
    --train_data data/train_qwen2.5-7b-it \
    --output_dir checkpoints/iap-qwen2.5-7b-it
```

### 3. Generation

```bash
# IAP
python generate.py --mode iap --model checkpoints/iap-qwen2.5-7b-it/final --chat \
    --output_dir outputs/qwen2.5-7b-it/iap

# Prompting baselines (untrained model)
python generate.py --mode implicit --model Qwen/Qwen2.5-7B-Instruct --chat \
    --output_dir outputs/qwen2.5-7b-it/implicit

# Explicit Intent Personalization / intent ablation (needs the IAP intents)
python generate.py --mode explicit --model Qwen/Qwen2.5-7B-Instruct --chat \
    --intents_dir outputs/qwen2.5-7b-it/iap --output_dir outputs/qwen2.5-7b-it/explicit
```


### 4. Evaluation

Final scores use the official LaMP-QA evaluator with Gemma-4-31B as the judge:

```bash
export EVAL_JUDGE=<path or HF id of Gemma-4-31B>
bash scripts/evaluate.sh outputs/qwen2.5-7b-it/iap
```
