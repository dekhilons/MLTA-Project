"""Pre-compute one wrong answer + faulty solution body per question -> data/faults_<split>.json."""
import argparse
import collections
import json

from _common import ROOT, load_config, make_llm
from mas.data import read_jsonl
from mas.faults import build_faults

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test"], required=True)
ap.add_argument("--config")
args = ap.parse_args()
cfg = load_config(args.config)

data_dir = ROOT / cfg["data"]["dir"]
items = read_jsonl(data_dir / f"{args.split}.jsonl")
method = cfg["faults"]["method"]
llm = make_llm(cfg) if method == "llm" else None
faults = build_faults(items, seed=cfg["seed"], method=method, llm=llm)

out = data_dir / f"faults_{args.split}.json"
out.write_text(json.dumps(faults, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", out, dict(collections.Counter(f["body_source"] for f in faults.values())))
