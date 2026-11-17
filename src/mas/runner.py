"""Experiment grid -> one JSONL log per condition (resumable: existing files are skipped)."""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from .agents import AggregatorAgent, FaultyAgent, SolverAgent, VerifierAgent
from .data import write_jsonl
from .topologies import run_team


@dataclass(frozen=True)
class Condition:
    topology: str                  # chain | star
    defense: str                   # none | skeptical | verifier
    fault_position: int | None     # None = clean baseline
    style: str | None              # neutral | persuasive | None

    @property
    def name(self) -> str:
        f = "clean" if self.fault_position is None else f"fault{self.fault_position}-{self.style}"
        return f"{self.topology}__{self.defense}__{f}"


def build_conditions(grid: dict) -> list[Condition]:
    conds = []
    for topo, dfn in itertools.product(grid["topologies"], grid["defenses"]):
        conds.append(Condition(topo, dfn, None, None))
        for pos, style in itertools.product(grid["fault_positions"], grid["styles"]):
            conds.append(Condition(topo, dfn, pos, style))
    return conds


def build_team(cond: Condition, faults: dict):
    skeptical = cond.defense == "skeptical"
    solvers = [FaultyAgent(f"Teammate {i+1}", faults, cond.style) if i == cond.fault_position
               else SolverAgent(f"Teammate {i+1}", skeptical=skeptical) for i in range(3)]
    verifier = VerifierAgent() if cond.defense == "verifier" else None
    return solvers, AggregatorAgent(skeptical=skeptical), verifier


def run_condition(cond: Condition, items: list[dict], faults: dict, llm, out_dir: Path) -> Path:
    path = Path(out_dir) / f"{cond.name}.jsonl"
    if path.exists():
        return path
    solvers, agg, verifier = build_team(cond, faults)
    records = run_team(items, cond.topology, solvers, agg, llm, verifier)
    for r in records:
        r.update(asdict(cond))
        r["injected_answer"] = faults[r["qid"]]["wrong"] if cond.fault_position is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, path)
    return path


def run_grid(conds: list[Condition], items, faults, llm, out_dir) -> None:
    for c in tqdm(conds, desc="conditions"):
        run_condition(c, items, faults, llm, out_dir)
