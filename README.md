# When One Agent Goes Wrong

**Error propagation and containment in small multi-agent LLM systems.**
MLTA 2026 project: Ons Dekhil, Slim Zarrouk, Abdullah Hameed.

---

## 1. What this project does

A team of small LLM agents (all **Qwen2.5-1.5B-Instruct**) solves **GSM8K** math word problems.
We replace one agent with a **scripted faulty agent** that always sends a wrong answer we chose
in advance, and measure:

- **Propagation:** do the other agents copy the wrong answer? Does it depend on the team
  structure (chain vs star) and on where the faulty agent sits?
- **Persuasion:** does a confident, persuasive faulty agent spread its error more than a neutral one?
- **Containment:** does a "be skeptical" prompt or a dedicated **verifier agent** stop the error,
  and what does each defense cost?

## 2. Architecture

The team is 3 solvers + 1 aggregator (the team lead who gives the final answer).

```
CHAIN: each solver sees the previous solver's solution and may revise it

  Question → [Solver 1] → [Solver 2] → [Solver 3] → [Aggregator] → final answer


STAR: solvers work independently; only the aggregator sees all of them

             ┌→ [Solver 1] ─┐
  Question ──┼→ [Solver 2] ─┼→ [Aggregator] → final answer
             └→ [Solver 3] ─┘
```

- **Fault injection:** one solver (position 0, 1 or 2) is replaced by a `FaultyAgent`. It does not
  call the model; it sends a pre-computed wrong solution in a **neutral** or **persuasive** style.
  Both styles use the same reasoning text, so the style is the only difference.
- **Defenses:**
  - `none`: no defense.
  - `skeptical`: every agent's system prompt adds "check your teammates' work before accepting it".
  - `verifier`: a verifier agent checks each solver message. Rejected messages are dropped and
    never reach the next solver or the aggregator.

**One experimental condition** = topology × defense × fault position × style.
The full grid is 2 topologies × 3 defenses × (1 clean baseline + 3 positions × 2 styles) = **42 conditions**.

### Pipeline

```
 scripts/prepare_data.py     GSM8K  ──► data/dev.jsonl (50 Qs, from train)
                                        data/test.jsonl (200 Qs, from test)
            │
            ▼
 scripts/make_faults.py      one wrong answer + faulty solution per question
                             ──► data/faults_<split>.json
            │
            ▼
 scripts/run_experiments.py  for every condition: run the team on every question
                             ──► results/logs/<split>/<condition>.jsonl  (one line per question,
                                 with every message, verdict, answer and token count)
            │
            ▼
 scripts/analyze.py          logs ──► results/tables/*.csv  +  results/figures/*.png
```

## 3. Project structure

```
MLTA-Project/
├── configs/
│   └── default.yaml        model, backend, split sizes, experiment grid. All settings live here.
├── prompts/                every prompt as a plain .txt file (edit prompts here, not in code)
│   ├── solver_*.txt            solver system prompt, solving alone / reviewing a peer
│   ├── aggregator_*.txt        team lead prompts
│   ├── verifier_*.txt          verifier prompt (must end with "VERDICT: ACCEPT/REJECT")
│   ├── skeptical_addon.txt     appended to system prompts for the "skeptical" defense
│   └── faulty_*.txt            how faulty solutions are generated and styled
├── src/mas/                the library (all logic lives here)
│   ├── llm.py                  loads the model and generates in batches (backend: hf or vllm),
│   │                           counts tokens; MockLLM for tests without a GPU
│   ├── data.py                 loads GSM8K, makes dev/test splits, extracts the final number
│   │                           from a model's text (####, \boxed{}, "answer is", last number)
│   ├── faults.py               picks a plausible wrong answer and builds the faulty solution
│   ├── agents.py               Message + SolverAgent, FaultyAgent, AggregatorAgent, VerifierAgent
│   ├── topologies.py           run_team(): chain / star orchestration, batched across questions
│   ├── runner.py               Condition, the 42-condition grid, writes one JSONL log per condition
│   └── metrics.py              accuracy, adoption, recovery, verifier precision/recall, cost
├── scripts/                command-line entry points (thin wrappers around src/mas)
│   ├── _common.py              config loading + model creation shared by the scripts
│   ├── prepare_data.py
│   ├── make_faults.py
│   ├── run_experiments.py
│   └── analyze.py
├── tests/                  unit tests, run on CPU with a fake model (no GPU, no download)
├── notebooks/              exploration + the final demo notebook (deliverable)
├── data/                   generated, gitignored (rebuild it with the scripts)
├── results/
│   ├── logs/                   raw JSONL logs (gitignored, large)
│   ├── tables/                 summary CSVs (committed, used in the report)
│   └── figures/                plots (committed, used in the report)
├── report/                 final report (deliverable)
└── docs/                   plan and design notes
```

## 4. Setup

```bash
pip install -r requirements.txt
```

**Running locally (Windows, RTX 4050 6 GB):** vLLM does not run on native Windows, so in
`configs/default.yaml` set:

```yaml
model:
  backend: hf
  dtype: bfloat16
```

**Running on Colab/Kaggle (Linux GPU):** `pip install vllm` and keep `backend: vllm`
(`float16` on a T4). This is faster for the full test run.

The model (~3 GB) downloads automatically on first use.

## 5. How to test

**Step 1. Unit tests (no GPU, a few seconds).** These check answer extraction, fault generation,
both topologies, the verifier and the metrics, using a fake model:

```bash
python -m pytest -q
```

**Step 2. Smoke test on the real model (a few minutes).** Run a handful of dev questions on a
single condition to check speed and output format:

```bash
python scripts/prepare_data.py
python scripts/make_faults.py --split dev
python scripts/run_experiments.py --split dev --limit 5 --only chain__none
python scripts/analyze.py --split dev_limit5
```

Then open a log in `results/logs/dev_limit5/` and read the messages yourself.

**Step 3. Dev runs (tuning).** Use the full dev split, iterate on `prompts/`, and re-run.
Logs are resumable: an existing log file is skipped, so delete it to re-run a condition.

```bash
python scripts/run_experiments.py --split dev
python scripts/analyze.py --split dev
```

**Step 4. Final test run (once).** When prompts are frozen:

```bash
python scripts/make_faults.py --split test
python scripts/run_experiments.py --split test --final
python scripts/analyze.py --split test
```

## 6. Metrics

| Metric | Definition |
|---|---|
| Team accuracy | final answer == gold answer |
| Error adoption rate | share of downstream agents whose answer == the injected wrong answer (chain: later solvers + aggregator; star: aggregator only) |
| Recovery rate | of the questions the clean team (same topology + defense) solves, the share still solved with the fault |
| Verifier precision / recall | positive = message with a wrong answer; predicted positive = rejected by the verifier |
| Cost | LLM calls and tokens per question |
