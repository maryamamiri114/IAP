"""Rubric-based LaMP-QA judge used as the personalized reward mu(x, y, E).

The judge prompt and JSON parsing are imported from the official LaMP-QA repo
(third_party/LaMP-QA, see scripts/setup.sh), so the training reward uses exactly
the benchmark's evaluation prompt. The judge model is served with vLLM's
OpenAI-compatible server, e.g.

    vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000

Scoring follows LaMP-QA: each rubric aspect gets a 0-2 match_score; the
question score is sum(scores) / (2 * #aspects), in [0, 1]. As in the official
evaluator, unparsable judge outputs are retried with temperature raised by 0.1
per retry, and scored 0 after `max_retries`.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor

_LAMPQA_DIR = os.environ.get(
    "LAMPQA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "LaMP-QA"),
)
if _LAMPQA_DIR not in sys.path:
    sys.path.insert(0, _LAMPQA_DIR)

try:
    from evaluation.evaluator import _EVAL_PROMPT_SYSTEM, _EVAL_PROMPT_USER, parse_json  # noqa: E402
except ImportError as e:  # pragma: no cover
    raise ImportError(
        f"Could not import the LaMP-QA evaluator from {_LAMPQA_DIR}. "
        "Run `bash scripts/setup.sh` or set LAMPQA_DIR."
    ) from e


def _format_aspect(aspect: dict) -> str:
    # Same formatting as create_eval_prompt() in the official evaluator.
    return f'-aspect: {aspect["aspect"]}\n    -reason: {aspect["reason"]}\n    -evidence: {aspect["evidence"]}'


class RubricJudge:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "Qwen/Qwen2.5-32B-Instruct",
                 api_key: str = "EMPTY", max_workers: int = 64, max_retries: int = 10,
                 max_tokens: int = 4096, top_p: float = 0.95, timeout: float = 600.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.top_p = top_p

    def _score_aspect(self, question: str, details: str, response: str, aspect: dict) -> int:
        messages = [
            {"role": "system", "content": _EVAL_PROMPT_SYSTEM},
            {"role": "user", "content": _EVAL_PROMPT_USER.format(
                question=question, details=details, response=response, aspects=_format_aspect(aspect))},
        ]
        temperature = 0.0
        for _ in range(self.max_retries + 1):
            out = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature,
                top_p=self.top_p, max_tokens=self.max_tokens,
            )
            try:
                score = parse_json(out.choices[0].message.content or "")["match_score"]
                return max(0, min(2, int(score)))
            except Exception:
                temperature = min(1.0, temperature + 0.1)
        return 0

    def score(self, question: str, details: str, response: str | None, aspects: list[dict]) -> float:
        """mu(x, y, E) in [0, 1]. A missing response scores 0 without calling the judge."""
        return self.score_batch([(question, details, response, aspects)])[0]

    def score_batch(self, items: list[tuple[str, str, str | None, list[dict]]]) -> list[float]:
        jobs, owners = [], []
        for i, (question, details, response, aspects) in enumerate(items):
            if not response:
                continue
            for aspect in aspects:
                jobs.append((question, details, response, aspect))
                owners.append(i)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            aspect_scores = list(pool.map(lambda j: self._score_aspect(*j), jobs))

        totals = [0] * len(items)
        for i, s in zip(owners, aspect_scores):
            totals[i] += s
        return [
            (totals[i] / (2 * len(aspects))) if (response and aspects) else 0.0
            for i, (_, _, response, aspects) in enumerate(items)
        ]
