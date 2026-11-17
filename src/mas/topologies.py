"""Chain and star orchestration, batched across all questions.

chain : solver1 -> solver2 -> solver3, each sees the previous (accepted) solution.
star  : solvers answer independently.
Both  : the aggregator sees all accepted solver messages and gives the team answer.
Verifier (optional) checks every solver message before it is forwarded; rejected
messages are dropped (not shown to the next solver nor to the aggregator).
"""
from __future__ import annotations

from dataclasses import asdict

from .agents import AggregatorAgent, Message, VerifierAgent


def _step(agent, items: list[dict], priors: list[list[Message]], llm) -> list[Message]:
    if not agent.is_llm:
        return [agent.respond(it) for it in items]
    gens = llm.generate([agent.chat(it, pr) for it, pr in zip(items, priors)])
    return [agent.parse(g) for g in gens]


def _verify(verifier: VerifierAgent | None, items: list[dict], msgs: list[Message], llm):
    if verifier is None:
        return
    gens = llm.generate([verifier.chat(it, m) for it, m in zip(items, msgs)])
    for m, g in zip(msgs, gens):
        verifier.apply(m, g)


def run_team(items: list[dict], topology: str, solvers: list, aggregator: AggregatorAgent,
             llm, verifier: VerifierAgent | None = None) -> list[dict]:
    n = len(items)
    history: list[list[Message]] = [[] for _ in range(n)]

    if topology == "chain":
        last_ok: list[Message | None] = [None] * n
        for agent in solvers:
            msgs = _step(agent, items, [[m] if m else [] for m in last_ok], llm)
            _verify(verifier, items, msgs, llm)
            for i, m in enumerate(msgs):
                history[i].append(m)
                if m.accepted:
                    last_ok[i] = m
    elif topology == "star":
        for agent in solvers:  # independent; could be one big batch, kept simple
            msgs = _step(agent, items, [[] for _ in range(n)], llm)
            _verify(verifier, items, msgs, llm)
            for i, m in enumerate(msgs):
                history[i].append(m)
    else:
        raise ValueError(topology)

    finals = _step(aggregator, items,
                   [[m for m in h if m.accepted] for h in history], llm)

    records = []
    for it, h, fin in zip(items, history, finals):
        all_msgs = h + [fin]
        records.append({
            "qid": it["qid"], "gold": it["gold"],
            "solvers": [asdict(m) for m in h],
            "aggregator": asdict(fin),
            "final_answer": fin.answer,
            "correct": fin.answer == it["gold"],
            "llm_calls": sum(m.llm_calls for m in all_msgs),
            "prompt_tokens": sum(m.prompt_tokens for m in all_msgs),
            "completion_tokens": sum(m.completion_tokens for m in all_msgs),
        })
    return records
