"""Pre-computed, reproducible faults.

For each question we fix ONE wrong answer and ONE solution body that reaches it.
The neutral and persuasive faulty messages share the exact same body, so the
style is the only difference between them (clean test of RQ2).
"""
from __future__ import annotations

import random
import re

from . import load_prompt
from .data import Number, extract_answer


def perturb_answer(gold: Number, rng: random.Random) -> tuple[Number, str]:
    """A plausible wrong answer (looks like a typical slip). Returns (wrong, error_type)."""
    if isinstance(gold, int):
        cands = [(gold + 1, "off_by_one"), (gold - 1, "off_by_one"),
                 (gold + 10, "carry"), (gold - 10, "carry"),
                 (gold * 2, "double"), (gold * 10, "magnitude")]
        if gold % 2 == 0:
            cands.append((gold // 2, "half"))
    else:
        cands = [(round(gold + 1, 4), "off_by_one"), (round(gold * 2, 4), "double"),
                 (round(gold / 2, 4), "half"), (round(gold * 10, 4), "magnitude")]
    cands = [(w, t) for w, t in cands if w != gold and w >= 0]
    return rng.choice(cands)


def corrupt_gold_solution(gold_solution: str, gold: Number, wrong: Number) -> str:
    """Deterministic fallback: GSM8K reference reasoning with the final result replaced."""
    body = gold_solution.split("####")[0]
    body = re.sub(r"<<[^>]*>>", "", body).strip()           # drop calculator annotations
    lines = body.splitlines()
    if lines:
        g = str(gold)
        last = lines[-1]
        idx = last.rfind(g)
        lines[-1] = (last[:idx] + str(wrong) + last[idx + len(g):]) if idx >= 0 \
            else f"{last} So the answer is {wrong}."
    return "\n".join(lines)


def strip_final_line(text: str) -> str:
    return text.split("####")[0].strip()


def build_faults(items: list[dict], seed: int, method: str = "llm", llm=None) -> dict:
    """Returns {qid: {wrong, error_type, body, body_source}}."""
    rng = random.Random(seed)
    faults: dict[str, dict] = {}
    for it in items:
        wrong, error_type = perturb_answer(it["gold"], rng)
        faults[it["qid"]] = {"wrong": wrong, "error_type": error_type}

    if method == "llm":
        assert llm is not None, "method='llm' needs an LLM"
        tmpl = load_prompt("faulty_generate")
        chats = [[{"role": "user", "content": tmpl.format(
            question=it["question"], wrong=faults[it["qid"]]["wrong"])}] for it in items]
        for it, gen in zip(items, llm.generate(chats)):
            f = faults[it["qid"]]
            if extract_answer(gen.text) == f["wrong"]:
                f["body"], f["body_source"] = strip_final_line(gen.text), "llm"
    for it in items:  # fallback (or primary if method == corrupt_gold)
        f = faults[it["qid"]]
        if "body" not in f:
            f["body"] = corrupt_gold_solution(it["gold_solution"], it["gold"], f["wrong"])
            f["body_source"] = "corrupt_gold"
    return faults


def render_faulty_message(fault: dict, style: str) -> str:
    return load_prompt(f"faulty_{style}").format(body=fault["body"], wrong=fault["wrong"])
