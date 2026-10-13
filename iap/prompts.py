"""Prompt templates.

The IAP, Implicit Intent Personalization and Explicit Intent Personalization
prompts are copied from the paper (Section 3.2 and Appendix B).

Prompts marked "NOT IN PAPER" are not printed in the paper. Replace them with
the exact text used in your experiments before releasing results.
"""

# --------------------------------------------------------------------------
# How the user question and its narrative are shown to the model (x_i).
# NOT IN PAPER: make sure this matches the formatting you actually used.
# --------------------------------------------------------------------------
QUESTION_TEMPLATE = "{question}\n\nDetails: {narrative}"


def format_question(question: str, narrative: str | None = None) -> str:
    """Builds x_i: the user question plus its narrative context."""
    if narrative:
        return QUESTION_TEMPLATE.format(question=question.strip(), narrative=narrative.strip())
    return question.strip()


# --------------------------------------------------------------------------
# IAP rollout prompt (paper, Section 3.2)
# --------------------------------------------------------------------------
IAP_PROMPT = (
    "Your task is to generate a personalized response to the user's question based on users' intent. "
    "To do this, you need to first think about the question and how to generate a personalized answer "
    "for the user based on the user intent and needs. The thinking process should be inside <think> and "
    "</think> tags. Then, generate the potential intent(s) as a numbered list enclosed within the "
    "<intent> and </intent> tags. Next, think again using those intent(s) to guide answer planning and "
    "generation, again inside <think> and </think> tags and put the final answers in the <answer> and "
    "</answer> tags. Nothing should be outside the mentioned tags except the initial question.\n"
    "Now, answer the following question: {question}"
)

# --------------------------------------------------------------------------
# Baselines (paper, Appendix B)
# --------------------------------------------------------------------------
IMPLICIT_INTENT_PROMPT = (
    "Your task is to generate a personalized response to the user's question. First, infer the user's "
    "intent, defined as the underlying goal, need, or reason behind asking the question. Then, use this "
    "understanding to produce a personalized response that is tailored to the user and directly addresses "
    "their needs. Think step by step. Now, answer the following question: {question}"
)

EXPLICIT_INTENT_PROMPT = (
    "Your task is to generate a personalized response to the user's question. Intent refers to the "
    "user's underlying goal, need, or reason for asking the question. Given the question and the user "
    "intent, generate a personalized response that directly addresses the user's needs. Think step by "
    "step. Now, answer the following question: {question}. User intent: {intent}"
)

# NOT IN PAPER: CoT no-personalization baseline (question only, no user info),
# following Salemi & Zamani (2025).
NON_PERSONALIZED_PROMPT = (
    "Answer the following question. Think step by step, then give your final answer. "
    "Question: {question}"
)

# NOT IN PAPER: prompt for the intent-free response y_bar ~ pi_ref(. | x_i)
# used by the contrastive reward. It sees the same x_i as the policy, but is
# not asked to infer or use any intent.
INTENT_FREE_PROMPT = (
    "Your task is to generate a response to the user's question. "
    "Now, answer the following question: {question}"
)

# NOT IN PAPER: personalized response without any intent step
# ("No Intent" condition of the intent ablation, Table 2).
NO_INTENT_PROMPT = (
    "Your task is to generate a personalized response to the user's question that directly addresses "
    "the user's needs. Think step by step. Now, answer the following question: {question}"
)

# NOT IN PAPER: approximation of SWI (Yin et al., 2025), where the model first
# states its intent explicitly and then answers. Replace with the prompt you used.
SWI_PROMPT = (
    "Before answering, explicitly state your intent: what you aim to accomplish with your response and "
    "how you plan to address the user's question. Then write your response.\n"
    "Question: {question}"
)


def build_prompt(mode: str, question: str, intent: str | None = None) -> str:
    """Returns the prompt for a generation mode. `question` is already formatted x_i."""
    if mode == "iap":
        return IAP_PROMPT.format(question=question)
    if mode == "implicit":
        return IMPLICIT_INTENT_PROMPT.format(question=question)
    if mode in ("explicit", "ingroup", "crossgroup"):
        if intent is None:
            raise ValueError(f"mode '{mode}' needs an intent")
        return EXPLICIT_INTENT_PROMPT.format(question=question, intent=intent)
    if mode == "nonpers":
        return NON_PERSONALIZED_PROMPT.format(question=question)
    if mode == "intent_free":
        return INTENT_FREE_PROMPT.format(question=question)
    if mode == "no_intent":
        return NO_INTENT_PROMPT.format(question=question)
    if mode == "swi":
        return SWI_PROMPT.format(question=question)
    raise ValueError(f"unknown mode: {mode}")
