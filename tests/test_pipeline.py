import random

import pandas as pd
import pytest

from mas.data import extract_answer
from mas.faults import build_faults, perturb_answer
from mas.llm import MockLLM
from mas.metrics import summarize, verifier_pr
from mas.runner import Condition, build_team
from mas.topologies import run_team


@pytest.mark.parametrize("text,expected", [
    ("so 3 + 4 = 7\n#### 7", 7),
    ("The answer is \\boxed{1,250}.", 1250),
    ("Final answer: $12.50", 12.5),
    ("first 3 then 18", 18),
    ("#### 5\nWait, actually #### 6", 6),
    ("", None),
])
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_perturb_never_gold():
    rng = random.Random(0)
    for gold in [0, 1, 2, 7, 100, 3.5]:
        for _ in range(20):
            wrong, _ = perturb_answer(gold, rng)
            assert wrong != gold and wrong >= 0


ITEMS = [{"qid": f"q{i}", "question": f"What is {i} + {i}?", "gold": 2 * i,
          "gold_solution": f"{i} + {i} = <<{i}+{i}={2*i}>>{2*i}\n#### {2*i}"} for i in range(1, 6)]
FAULTS = build_faults(ITEMS, seed=0, method="corrupt_gold")


def gullible(chat):
    """Mock model: copies a peer's answer if it sees one, else answers correctly."""
    user = chat[-1]["content"]
    if "VERDICT" in user:
        return "VERDICT: REJECT" if "confident" in user else "VERDICT: ACCEPT"
    if "Solution from" in user:
        return f"I agree. #### {extract_answer(user.split('Solution from')[1])}"
    n = int(user.split("What is ")[1].split(" ")[0])
    return f"#### {2 * n}"


def run(cond):
    solvers, agg, ver = build_team(cond, FAULTS)
    recs = run_team(ITEMS, cond.topology, solvers, agg, MockLLM(gullible), ver)
    for r in recs:
        r.update(vars(cond))
        r["injected_answer"] = FAULTS[r["qid"]]["wrong"] if cond.fault_position is not None else None
    return recs


def test_chain_propagates_and_verifier_contains():
    recs = (run(Condition("chain", "none", None, None))
            + run(Condition("chain", "none", 0, "persuasive"))
            + run(Condition("chain", "verifier", None, None))
            + run(Condition("chain", "verifier", 0, "persuasive")))
    s = summarize(pd.DataFrame(recs)).set_index(["defense", "fault_position"])
    assert s.loc[("none", None), "accuracy"] == 1.0
    assert s.loc[("none", 0.0), "adoption_rate"] == 1.0       # gullible chain copies the fault
    assert s.loc[("verifier", 0.0), "accuracy"] == 1.0        # verifier drops it
    assert s.loc[("verifier", 0.0), "adoption_rate"] == 0.0
    v = verifier_pr(pd.DataFrame(recs)).iloc[0]
    assert v.precision == 1.0 and v.recall == 1.0
