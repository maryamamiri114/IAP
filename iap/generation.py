"""Offline generation with vLLM (inference, baselines, intent-free answers, SFT sampling)."""

from __future__ import annotations


def load_llm(model: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9,
             max_model_len: int = 8192, seed: int = 42):
    from vllm import LLM

    return LLM(model=model, tensor_parallel_size=tensor_parallel_size,
               gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, seed=seed)


def generate(llm, prompts: list[str], use_chat_template: bool, temperature: float, top_p: float,
             max_tokens: int, n: int = 1, seed: int | None = None) -> list[list[str]]:
    """Returns n completions per prompt.

    use_chat_template=True wraps each prompt as a single user turn (instruction-tuned
    models); False feeds the raw prompt (base models).
    """
    from vllm import SamplingParams

    params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n, seed=seed)
    if use_chat_template:
        outputs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)
    else:
        outputs = llm.generate(prompts, params)
    return [[o.text for o in out.outputs] for out in outputs]
