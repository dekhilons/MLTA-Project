"""Aggregate logs into tables + figures for the report."""
import argparse

import matplotlib.pyplot as plt

from _common import ROOT, load_config
from mas.metrics import load_logs, summarize, verifier_pr

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", help="log folder name, e.g. dev, test, dev_limit20")
ap.add_argument("--config")
args = ap.parse_args()
cfg = load_config(args.config)

res = ROOT / cfg["results_dir"]
df = load_logs(res / "logs" / args.split)
summary, vpr = summarize(df), verifier_pr(df)
(res / "tables").mkdir(parents=True, exist_ok=True)
summary.to_csv(res / "tables" / f"summary_{args.split}.csv", index=False)
vpr.to_csv(res / "tables" / f"verifier_{args.split}.csv", index=False)
print(summary.round(3).to_string(index=False))
print(vpr.round(3).to_string(index=False))

# RQ1: adoption by topology x fault position (no defense, neutral)
fig_dir = res / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
s = summary[(summary.defense == "none") & (summary["style"] == "neutral")]
if not s.empty:
    ax = s.pivot(index="fault_position", columns="topology", values="adoption_rate").plot.bar()
    ax.set(ylabel="error adoption rate", xlabel="faulty agent position", ylim=(0, 1))
    plt.tight_layout(); plt.savefig(fig_dir / f"rq1_adoption_{args.split}.png", dpi=150); plt.close()

# RQ2: neutral vs persuasive
s = summary[(summary.defense == "none") & summary["style"].notna()]
if not s.empty:
    ax = s.groupby(["topology", "style"])["adoption_rate"].mean().unstack().plot.bar()
    ax.set(ylabel="error adoption rate", ylim=(0, 1))
    plt.tight_layout(); plt.savefig(fig_dir / f"rq2_style_{args.split}.png", dpi=150); plt.close()

# RQ3: accuracy vs cost per defense
s = summary[summary["style"].notna()].groupby(["topology", "defense"])[
    ["accuracy", "tokens_per_q"]].mean().reset_index()
if not s.empty:
    fig, ax = plt.subplots()
    for topo, grp in s.groupby("topology"):
        ax.scatter(grp.tokens_per_q, grp.accuracy, label=topo)
        for _, r in grp.iterrows():
            ax.annotate(r.defense, (r.tokens_per_q, r.accuracy))
    ax.set(xlabel="tokens per question", ylabel="team accuracy (faulted runs)"); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir / f"rq3_defense_cost_{args.split}.png", dpi=150); plt.close()
