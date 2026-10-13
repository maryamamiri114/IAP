"""IAP reward (paper, Section 3.3):

    r = lambda1 * r_pers - lambda2 * r_contrast - lambda3 * r_length

    r_pers     = mu(x, y_hat, E)                                   (answer tag only)
    r_contrast = lambda * BERTScore(y_hat, y_bar) + max(0, mu(y_bar) - mu(y_hat))
    r_length   = max(0, |<intent>| - L)                            (tokens)

The three terms are exposed as separate TRL reward functions; the weights
(lambda1, -lambda2, -lambda3) are passed as `reward_weights`, so TRL sums them
and logs each component separately.

The intent-free answer y_bar and its score mu(y_bar) are precomputed with the
frozen reference policy (scripts/prepare_intent_free.py) and arrive here as
the dataset columns `intent_free_answer` and `intent_free_score`.

Edge cases (NOT specified in the paper; documented in the README):
  * no <answer> tag  -> r_pers = 0, BERTScore term = 1 (maximum similarity penalty)
  * no <intent> tag  -> r_length = 0 (the answer is still scored if present)
"""

from __future__ import annotations

import os

from .judge import RubricJudge
from .parsing import completion_to_text, extract_tag


class IAPRewards:
    def __init__(self, judge: RubricJudge, tokenizer, lam: float = 0.01, intent_len_threshold: int = 400,
                 bertscore_model: str = "microsoft/deberta-v3-large", bertscore_layers: int = 18,
                 bertscore_batch_size: int = 32):
        self.judge = judge
        self.tokenizer = tokenizer
        self.lam = lam
        self.L = intent_len_threshold
        self.bertscore_model = bertscore_model
        self.bertscore_layers = bertscore_layers
        self.bertscore_batch_size = bertscore_batch_size
        self._scorer = None
        self._mu_cache: dict[tuple[str, str], float] = {}

    # ---------------------------------------------------------------- helpers
    def _bertscorer(self):
        if self._scorer is None:
            import torch
            from bert_score import BERTScorer

            device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu"
            self._scorer = BERTScorer(model_type=self.bertscore_model, num_layers=self.bertscore_layers,
                                      batch_size=self.bertscore_batch_size, device=device)
        return self._scorer

    @staticmethod
    def _answers(completions) -> list[str | None]:
        return [extract_tag(completion_to_text(c), "answer") for c in completions]

    def _mu(self, uids, questions, narratives, aspects, answers) -> list[float]:
        """Judge scores for answers, cached by (uid, answer) so r_pers and r_contrast share calls."""
        todo = [i for i, (u, a) in enumerate(zip(uids, answers)) if a and (u, a) not in self._mu_cache]
        if todo:
            scores = self.judge.score_batch(
                [(questions[i], narratives[i], answers[i], aspects[i]) for i in todo])
            for i, s in zip(todo, scores):
                self._mu_cache[(uids[i], answers[i])] = s
        return [self._mu_cache[(u, a)] if a else 0.0 for u, a in zip(uids, answers)]

    # ---------------------------------------------------------- reward funcs
    def r_pers(self, prompts, completions, uid, question, narrative, rubric_aspects, **kwargs) -> list[float]:
        self._mu_cache.clear()  # r_pers runs first in each step; reset the per-step cache
        return self._mu(uid, question, narrative, rubric_aspects, self._answers(completions))

    def r_contrast(self, prompts, completions, uid, question, narrative, rubric_aspects,
                   intent_free_answer, intent_free_score, **kwargs) -> list[float]:
        answers = self._answers(completions)
        mu_hat = self._mu(uid, question, narrative, rubric_aspects, answers)

        sim = [1.0] * len(answers)
        idx = [i for i, a in enumerate(answers) if a]
        if idx:
            _, _, f1 = self._bertscorer().score([answers[i] for i in idx],
                                                [intent_free_answer[i] for i in idx])
            for i, v in zip(idx, f1.tolist()):
                sim[i] = float(v)

        return [self.lam * s + max(0.0, float(mb) - mh)
                for s, mb, mh in zip(sim, intent_free_score, mu_hat)]

    def r_length(self, prompts, completions, **kwargs) -> list[float]:
        out = []
        for c in completions:
            intent = extract_tag(completion_to_text(c), "intent")
            n = len(self.tokenizer(intent, add_special_tokens=False)["input_ids"]) if intent else 0
            out.append(float(max(0, n - self.L)))
        return out
