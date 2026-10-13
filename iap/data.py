"""LaMP-QA loading (https://huggingface.co/datasets/alireza7/LaMP-QA)."""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets, load_dataset

HF_DATASET = "alireza7/LaMP-QA"

DOMAINS = {
    "art": "Art_and_Entertainment",
    "lifestyle": "Lifestyle_and_Personal_Development",
    "society": "Society_and_Culture",
}

KEEP_COLUMNS = ["id", "question", "narrative", "rubric_aspects", "category"]


def load_domain(domain: str, split: str, cache_dir: str | None = None,
                max_samples: int | None = None) -> Dataset:
    """Loads one domain/split. Adds `domain` and a globally unique `uid`.

    The `profile` column (user history) is dropped: IAP does not use it.
    Question ids are only unique within a domain, hence `uid`.
    """
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {list(DOMAINS)}")
    ds = load_dataset(HF_DATASET, DOMAINS[domain], split=split, cache_dir=cache_dir)
    ds = ds.remove_columns([c for c in ds.column_names if c not in KEEP_COLUMNS])
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.map(lambda ex: {"domain": domain, "uid": f"{domain}:{ex['id']}"})
    return ds


def load_domains(domains: list[str], split: str, cache_dir: str | None = None,
                 max_samples: int | None = None) -> Dataset:
    parts = [load_domain(d, split, cache_dir, max_samples) for d in domains]
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]
