"""Parsing of tagged IAP rollouts: <think> ... <intent> ... <think> ... <answer>."""

from __future__ import annotations

import re

_TAG_RE = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.S) for t in ("think", "intent", "answer")}
_FULL_FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<intent>.*?</intent>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
    re.S,
)
_NUMBERED_ITEM_RE = re.compile(r"^\s*\d+[\.\)]\s+\S", re.M)


def extract_tag(text: str, tag: str) -> str | None:
    """Content of the last <tag>...</tag> block, or None if absent or empty."""
    matches = _TAG_RE[tag].findall(text or "")
    if not matches:
        return None
    content = matches[-1].strip()
    return content or None


def is_well_formed(text: str) -> bool:
    """True if the rollout follows think -> intent -> think -> answer exactly."""
    return bool(_FULL_FORMAT_RE.match(text or ""))


def parse_rollout(text: str) -> dict:
    return {
        "intent": extract_tag(text, "intent"),
        "answer": extract_tag(text, "answer"),
        "well_formed": is_well_formed(text),
    }


def count_intents(intent_text: str | None) -> int:
    """Number of items in the numbered intent list (used for the Low/Medium/High analysis)."""
    if not intent_text:
        return 0
    n = len(_NUMBERED_ITEM_RE.findall(intent_text))
    return n if n > 0 else 1


def completion_to_text(completion) -> str:
    """TRL passes completions as strings (standard) or message lists (conversational)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return str(completion)
