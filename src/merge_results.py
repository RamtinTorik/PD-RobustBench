"""Merge per-dtype part CSVs (parallel runs) into canonical result files.

Usage:
  python merge_results.py --parts ../results/parts/s42 --tag ""        # seed 42 (canonical)
  python merge_results.py --parts ../results/parts/s43 --tag "_s43"

Produces (in results/):
  robustness_results{tag}.csv, predictions{tag}.csv, xai_similarity{tag}.csv
Integrity check: every (dataset, model) must have 1 clean + 8x5 corrupted rows.
"""
import argparse
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, CORRUPTION_LEVELS

N_CONDS = 1 + sum(len(v) for v in CORRUPTION_LEVELS.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(RESULTS_DIR / "parts"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    parts = Path(args.parts)

    res = [pd.read_csv(f) for f in sorted(parts.glob("robustness_*.csv"))]
    out = pd.concat(res, ignore_index=True)
    out.to_csv(RESULTS_DIR / f"robustness_results{args.tag}.csv", index=False)

    preds = [pd.read_csv(f) for f in sorted(parts.glob("predictions_*.csv"))]
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(
            RESULTS_DIR / f"predictions{args.tag}.csv", index=False)

    xais = [pd.read_csv(f) for f in sorted(parts.glob("xai_*.csv"))]
    if xais:
        pd.concat(xais, ignore_index=True).to_csv(
            RESULTS_DIR / f"xai_similarity{args.tag}.csv", index=False)

    chk = out.groupby(["dataset", "model"]).size()
    assert (chk == N_CONDS).all(), f"missing conditions (need {N_CONDS}):\n{chk}"
    print(f"[merge{args.tag}] robustness rows: {len(out)} | "
          f"datasets: {sorted(out.dataset.unique())} | models: {sorted(out.model.unique())} "
          f"| integrity OK ({N_CONDS} conditions per config)")
    if preds:
        print(f"[merge{args.tag}] prediction rows: {sum(len(p) for p in preds)}")
    if xais:
        print(f"[merge{args.tag}] xai rows: {sum(len(x) for x in xais)}")


if __name__ == "__main__":
    main()
