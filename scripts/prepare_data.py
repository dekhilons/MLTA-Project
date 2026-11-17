"""Create data/dev.jsonl (GSM8K train sample) and data/test.jsonl (GSM8K test sample)."""
import argparse

from _common import ROOT, load_config
from mas.data import make_splits

ap = argparse.ArgumentParser()
ap.add_argument("--config")
args = ap.parse_args()
cfg = load_config(args.config)
make_splits(cfg["data"]["n_dev"], cfg["data"]["n_test"], cfg["seed"], ROOT / cfg["data"]["dir"])
print("wrote dev/test splits to", ROOT / cfg["data"]["dir"])
