"""GSM8K loading, dev/test splits, and numeric answer extraction."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

Number = int | float

_NUM = r"(-?[\d,]*\.?\d+)"
_PATTERNS = [
    re.compile(r"####\s*\$?\s*" + _NUM),
    re.compile(r"\\boxed\{\s*\$?\s*" + _NUM),
    re.compile(r"final answer(?: is)?\s*:?\s*\$?\s*" + _NUM, re.IGNORECASE),
    re.compile(r"answer is\s*:?\s*\$?\s*" + _NUM, re.IGNORECASE),
]
_ANY_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize_number(s: str | None) -> Number | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        x = float(s)
    except ValueError:
        return None
    return int(x) if x.is_integer() else round(x, 4)


def extract_answer(text: str | None) -> Number | None:
    """Prefer explicit answer markers (last occurrence); fall back to the last number."""
    if not text:
        return None
    for pat in _PATTERNS:
        found = pat.findall(text)
        if found:
            return normalize_number(found[-1])
    nums = _ANY_NUM.findall(text)
    return normalize_number(nums[-1]) if nums else None


def load_gsm8k(split: str) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"qid": f"{split}-{i}", "question": r["question"],
             "gold": normalize_number(r["answer"].split("####")[-1]),
             "gold_solution": r["answer"]} for i, r in enumerate(ds)]


def make_splits(n_dev: int, n_test: int, seed: int, out_dir: str | Path) -> None:
    """dev <- GSM8K train (tuning only); test <- GSM8K test (used once)."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(rng.sample(load_gsm8k("train"), n_dev), out_dir / "dev.jsonl")
    write_jsonl(rng.sample(load_gsm8k("test"), n_test), out_dir / "test.jsonl")


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
