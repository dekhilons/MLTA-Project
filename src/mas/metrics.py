"""Metrics computed from the JSONL logs.

- accuracy            : final team answer == gold
- adoption rate       : share of DOWNSTREAM agents whose answer == injected answer
                        chain -> solvers after the faulty one + aggregator
                        star  -> aggregator only (solvers never see each other)
- final adoption      : final team answer == injected answer
- recovery rate       : among questions the CLEAN team (same topology+defense) solves,
                        share still solved with the fault injected
- verifier P/R        : positive = message with a wrong answer; predicted positive = rejected
- cost                : LLM calls and tokens per question
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import read_jsonl

KEYS = ["topology", "defense", "fault_position", "style"]


def load_logs(log_dir: str | Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(log_dir).glob("*.jsonl")):
        rows.extend(read_jsonl(p))
    return pd.DataFrame(rows)


def downstream_answers(rec: dict) -> list:
    pos = rec["fault_position"]
    if pos is None or pd.isna(pos):
        return []
    pos = int(pos)
    down = rec["solvers"][pos + 1:] if rec["topology"] == "chain" else []
    return [m["answer"] for m in down] + [rec["final_answer"]]


def add_record_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["faulted"] = df["fault_position"].notna()
    df["adoption"] = [
        (sum(a == r["injected_answer"] for a in d) / len(d)) if (d := downstream_answers(r)) else None
        for r in df.to_dict("records")]
    df["final_adopted"] = df["faulted"] & (df["final_answer"] == df["injected_answer"])
    df["total_tokens"] = df["prompt_tokens"] + df["completion_tokens"]
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_record_metrics(df)
    clean = df[~df["faulted"]][["topology", "defense", "qid", "correct"]] \
        .rename(columns={"correct": "clean_correct"})
    df = df.merge(clean, on=["topology", "defense", "qid"], how="left")
    df["recovered"] = df["correct"].where(df["clean_correct"] == True)  # noqa: E712

    g = df.groupby(KEYS, dropna=False)
    out = pd.DataFrame({
        "n": g.size(),
        "accuracy": g["correct"].mean(),
        "adoption_rate": g["adoption"].mean(),
        "final_adoption": g["final_adopted"].mean(),
        "recovery_rate": g["recovered"].mean(),
        "llm_calls_per_q": g["llm_calls"].mean(),
        "tokens_per_q": g["total_tokens"].mean(),
    }).reset_index()
    return out


def verifier_pr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df[df["defense"] == "verifier"].to_dict("records"):
        for m in r["solvers"]:
            rows.append({"topology": r["topology"], "faulty": m["faulty"],
                         "wrong": m["answer"] != r["gold"], "rejected": not m["accepted"]})
    m = pd.DataFrame(rows)
    if m.empty:
        return m

    def pr(x: pd.DataFrame) -> pd.Series:
        tp = (x.wrong & x.rejected).sum()
        return pd.Series({"precision": tp / max(x.rejected.sum(), 1),
                          "recall": tp / max(x.wrong.sum(), 1),
                          "recall_on_injected": x[x.faulty].rejected.mean(),
                          "false_reject_rate": x[~x.wrong].rejected.mean()})
    return m.groupby("topology").apply(pr, include_groups=False).reset_index()
