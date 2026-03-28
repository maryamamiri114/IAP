## Title
**Training LLMs with Reinforcement Learning for Intent-Aware Personalized Question Answering**

## Abstract
Personalized question answering can be substantially advanced by aligning answers with the user's underlying intent, the implicit ``why'' that drives a query beyond what is explicitly asked. However, most existing intent modeling approaches rely on closed-domain taxonomies or surface-level cues, and are typically evaluated in settings that assume access to multi-turn conversational context or rich user profiles. As a result, these methods struggle in single-turn scenarios, where the user's latent goal must be inferred from minimal textual input alone. To address this gap, we propose IAP (\textbf{I}ntent \textbf{A}ware \textbf{P}ersonalization), a reinforcement learning framework that trains models to infer implicit user intent directly from a single-turn question and incorporate it into their reasoning process to generate personalized, intent-grounded answers. By optimizing intent-aware answer trajectories under a personalized reward function, IAP reinforces generation paths that make implicit user intent explicit and produce responses that better align with the user's underlying goal and personalized needs as captured by the reward model. Through extensive experiments on the LaMP-QA benchmark across six models, IAP consistently outperforms all baselines, achieving an average macro-score gain of approximately 7.5\% over the strongest competitor, demonstrating that modeling implicit user intent within the training objective is a promising direction for PQA.


## Installing requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation
You need to download the [LaMP-QA](https://github.com/LaMP-Benchmark/LaMP-QA?tab=readme-ov-file#downloading-the-dataset) dataset. For this purpose, you can use the following code:

```bash
python download.py \
    --dataset_save_directory /path/to/download_directory \
    --cache_dir /path/to/cache_directory
```

## Running IAP

```bash
python IAP.py \
    --questions_address /path/to/prepared_dataset.jsonl \
    --output_address /path/to/output_directory \
    --model <qwen|gemma> \
    --temperature <temp> \
    --judge-model <judge> \
    --split test \
    --subsets ALL \
    --max-model-len <max_len>
```

## Evaluation

Please use the evaluation script provided by the [LaMP-QA benchmark](https://github.com/LaMP-Benchmark/LaMP-QA?tab=readme-ov-file#evaluating-the-generated-responses) to evaluate generated responses.


