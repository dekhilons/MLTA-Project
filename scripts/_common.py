import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402


def load_config(path: str | None = None) -> dict:
    with open(path or ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_llm(cfg: dict):
    from mas.llm import LLM
    m = cfg["model"]
    return LLM(m["name"], backend=m["backend"], dtype=m["dtype"],
               max_new_tokens=m["max_new_tokens"], temperature=m["temperature"],
               seed=cfg["seed"])
