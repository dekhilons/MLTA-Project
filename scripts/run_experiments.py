"""Run the condition grid. Logs -> results/logs/<split>/<condition>.jsonl (resumable)."""
import argparse
import json
import sys

from _common import ROOT, load_config, make_llm
from mas.data import read_jsonl
from mas.runner import build_conditions, run_grid

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test"], default="dev")
ap.add_argument("--final", action="store_true", help="required to touch the test split")
ap.add_argument("--limit", type=int, help="only the first N questions (smoke tests)")
ap.add_argument("--only", help="substring filter on condition names, e.g. 'chain__none'")
ap.add_argument("--config")
args = ap.parse_args()

if args.split == "test" and not args.final:
    sys.exit("The test split is used ONCE. Re-run with --final when prompts are frozen.")

cfg = load_config(args.config)
data_dir = ROOT / cfg["data"]["dir"]
items = read_jsonl(data_dir / f"{args.split}.jsonl")[: args.limit]
faults = json.loads((data_dir / f"faults_{args.split}.json").read_text(encoding="utf-8"))

conds = build_conditions(cfg["grid"])
if args.only:
    conds = [c for c in conds if args.only in c.name]
tag = args.split if args.limit is None else f"{args.split}_limit{args.limit}"
out_dir = ROOT / cfg["results_dir"] / "logs" / tag

print(f"{len(conds)} conditions x {len(items)} questions -> {out_dir}")
run_grid(conds, items, faults, make_llm(cfg), out_dir)
