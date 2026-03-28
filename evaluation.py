"""
evaluation.py
-------------
Evaluates IAP-generated responses using the LaMP-QA rubric evaluator.

The evaluator uses the official evaluator.py logic from:
  https://github.com/LaMP-Benchmark/LaMP-QA/blob/main/evaluation/evaluator.py

It scores each generated answer against the per-user rubric aspects supplied
in the dataset and reports the mean personalized score.

Usage
-----
python evaluation.py \
    --evaluator_model  Qwen/Qwen2.5-32B-Instruct \
    --dataset_path     data/validation.jsonl \
    --response_path    outputs/validation_responses.json \
    --score_path       outputs/validation_scores.json \
    [--tensor_parallel_size 1]

Response file format (same as LaMP-QA repo):
{
    "<question_id>": [{"output": "<generated answer>"}],
    ...
}

Score output format:
{
    "mean_score": float,
    "per_question_scores": [
        {"id": ..., "score": float, "details": [{"aspect": ..., "score": int}]},
        ...
    ]
}
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Inline copy of the official LaMP-QA evaluator logic
# (source: https://github.com/LaMP-Benchmark/LaMP-QA/blob/main/evaluation/evaluator.py)
# ---------------------------------------------------------------------------
import json5
from vllm import LLM, SamplingParams

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


def _parse_json(json_str: str):
    json_str = json_str.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(json_str, strict=False)
    except Exception:
        pass
    try:
        return json5.loads(json_str)
    except Exception:
        print(json_str)
        raise ValueError("Invalid json object")


def _create_eval_prompt(question, details, response, aspect, tokenizer):
    aspect_str = (
        f'-aspect: {aspect["aspect"]}\n'
        f' -reason: {aspect["reason"]}\n'
        f' -evidence: {aspect["evidence"]}'
    )
    conversation = [
        {"role": "system", "content": _EVAL_PROMPT_SYSTEM},
        {
            "role": "user",
            "content": _EVAL_PROMPT_USER.format(
                question=question,
                details=details,
                response=response,
                aspects=aspect_str,
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )


def _create_eval_prompts_all(queries, responses, details_list, aspects_list, tokenizer):
    prompts, ids = [], []
    for i, (query, response, detail, aspect) in enumerate(
        zip(queries, responses, details_list, aspects_list)
    ):
        for j, asp in enumerate(aspect):
            prompt = _create_eval_prompt(query, detail, response, asp, tokenizer)
            prompts.append(prompt)
            ids.append({"q_id": i, "a_id": j})
    return ids, prompts


def lamp_qa_evaluator(queries, responses, details_list, aspects_list, llm,
                      max_retries: int = 100):
    """
    Official LaMP-QA evaluator function.
    Returns a dict with keys: score (float), per_question_scores (list).
    """
    temperature = 0.0
    tokenizer = llm.get_tokenizer()
    retries = 0

    ids, prompts = _create_eval_prompts_all(
        queries, responses, details_list, aspects_list, tokenizer
    )
    outputs_dict: dict = {}

    while prompts:
        retries += 1
        sampling_params = SamplingParams(
            temperature=temperature, top_p=0.95, max_tokens=4096, logprobs=1
        )
        outputs = llm.generate(prompts, sampling_params)
        wrongs = []

        for id_, prompt, output in zip(ids, prompts, outputs):
            q_id = id_["q_id"]
            a_id = id_["a_id"]
            if q_id not in outputs_dict:
                outputs_dict[q_id] = {}
            try:
                obj = _parse_json(output.outputs[0].text)
                _ = obj["match_score"]          # validate key exists
                outputs_dict[q_id][a_id] = obj
            except Exception as e:
                print(e)
                if retries > max_retries:
                    outputs_dict[q_id][a_id] = {"match_score": 0}
                    continue
                wrongs.append((id_, prompt))

        prompts = [p for _, p in wrongs]
        ids     = [i for i, _ in wrongs]
        if temperature < 1.0:
            temperature += 0.1

    scores = []
    for i, (query, aspect) in enumerate(zip(queries, aspects_list)):
        score_query = 0
        detail_list = []
        for j, asp in enumerate(aspect):
            score_query += outputs_dict[i][j]["match_score"]
            detail_list.append(
                {"aspect": asp["aspect"], "score": outputs_dict[i][j]["match_score"]}
            )
        scores.append(
            {
                "id": i,
                "score": score_query / (len(aspect) * 2),
                "details": detail_list,
            }
        )

    avg = {
        "score": sum(s["score"] for s in scores) / len(scores),
        "per_question_scores": scores,
    }
    return avg


# ---------------------------------------------------------------------------
# Main evaluation driver
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IAP-generated responses with the LaMP-QA rubric evaluator."
    )
    parser.add_argument(
        "--evaluator_model", type=str, default="Qwen/Qwen2.5-32B-Instruct",
        help="Model used as the rubric judge (default: Qwen/Qwen2.5-32B-Instruct)."
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to the processed JSONL dataset file (e.g. data/validation.jsonl)."
    )
    parser.add_argument(
        "--response_path", type=str, required=True,
        help="Path to the JSON file with generated responses."
    )
    parser.add_argument(
        "--score_path", type=str, required=True,
        help="Path to write the JSON score output."
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs for tensor parallelism in vLLM."
    )
    parser.add_argument(
        "--max_retries", type=int, default=100,
        help="Max retries for failed JSON parses in the evaluator."
    )
    args = parser.parse_args()

    # --- Load dataset -------------------------------------------------------
    print(f"Loading dataset from {args.dataset_path} …")
    dataset = load_jsonl(args.dataset_path)
    id_to_record = {r["id"]: r for r in dataset}

    # --- Load responses -----------------------------------------------------
    print(f"Loading responses from {args.response_path} …")
    with open(args.response_path, encoding="utf-8") as f:
        responses_raw = json.load(f)
    # responses_raw: { question_id: [{"output": "..."}] }

    # Align responses with dataset records (preserve order)
    queries, responses, details_list, aspects_list = [], [], [], []
    missing = 0
    for qid, resp_list in responses_raw.items():
        if qid not in id_to_record:
            print(f"WARNING: question id '{qid}' not found in dataset, skipping.")
            missing += 1
            continue
        rec = id_to_record[qid]
        queries.append(rec["question"])
        details_list.append(rec.get("details", ""))
        aspects_list.append(rec["aspects"])
        # Take the first output in the list
        responses.append(resp_list[0]["output"])

    print(f"Evaluating {len(queries)} responses ({missing} skipped) …")
    if not queries:
        print("No valid responses to evaluate. Exiting.")
        sys.exit(1)

    # --- Load evaluator LLM -------------------------------------------------
    print(f"Loading evaluator model: {args.evaluator_model} …")
    llm = LLM(
        model=args.evaluator_model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )

    # --- Run evaluation -----------------------------------------------------
    result = lamp_qa_evaluator(
        queries=queries,
        responses=responses,
        details_list=details_list,
        aspects_list=aspects_list,
        llm=llm,
        max_retries=args.max_retries,
    )

    print(f"\n=== Evaluation Results ===")
    print(f"Mean personalized score: {result['score']:.4f}")

    # --- Save scores --------------------------------------------------------
    os.makedirs(os.path.dirname(args.score_path) or ".", exist_ok=True)
    with open(args.score_path, "w", encoding="utf-8") as f:
        json.dump(
            {"mean_score": result["score"], "per_question_scores": result["per_question_scores"]},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Scores saved to {args.score_path}")


if __name__ == "__main__":
    main()
