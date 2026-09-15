"""Parkinson Robustness Score (PRS) - task 6.1.

For every model and dataset type, quantifies how much the model degrades under
the corruption benchmark, combining three ingredients per corruption family c:

    dAcc_c  = mean over levels of relative drop in *balanced accuracy*
              (balanced accuracy is used because HandPD is ~4:1 imbalanced;
               for the balanced PDRAW sets it equals plain accuracy)
    dF1_c   = mean over levels of relative drop in F1 (PD class)
    dXAI_c  = mean over levels of (1 - Spearman corr between the corrupted
              Grad-CAM map and the clean Grad-CAM map)  [XAI instability]

    D_c     = w_acc * dAcc_c + w_f1 * dF1_c + w_xai * dXAI_c      (w = .5/.3/.2)

    PRS     = clip(100 * (1 - mean_c D_c), 0, 100)   (higher = more robust)

Outputs:
    results/robustness_scores{tag}.csv       (PRS + per-term contributions)
    results/robustness_breakdown{tag}.csv    (per-corruption D_c for every model)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import RESULTS_DIR, W_ACC, W_F1, W_XAI

RESULTS_CSV = RESULTS_DIR / "robustness_results.csv"
XAI_CSV = RESULTS_DIR / "xai_similarity.csv"
OUT_SCORES = RESULTS_DIR / "robustness_scores.csv"
OUT_BREAKDOWN = RESULTS_DIR / "robustness_breakdown.csv"


def relative_drop(clean_val: float, corr_val: float) -> float:
    """Relative degradation w.r.t. the clean value (can be negative = gain)."""
    if clean_val <= 0:
        return 0.0
    return (clean_val - corr_val) / clean_val


def compute(results_csv=RESULTS_CSV, xai_csv=XAI_CSV,
            out_scores=OUT_SCORES, out_breakdown=OUT_BREAKDOWN, tag=""):
    df = pd.read_csv(results_csv)
    clean = df[df.condition_type == "clean"].set_index(["dataset", "model"])
    corr = df[df.condition_type == "corrupted"].copy()

    xa = pd.read_csv(xai_csv) if xai_csv.exists() else None

    rows, breakdown = [], []
    for (dtype, model), g in corr.groupby(["dataset", "model"]):
        c = clean.loc[(dtype, model)]
        per_corr = []
        for cname, gc in g.groupby("corruption"):
            d_acc = gc.apply(lambda r: relative_drop(c.balanced_accuracy,
                                                     r.balanced_accuracy), axis=1).mean()
            d_f1 = gc.apply(lambda r: relative_drop(c.f1, r.f1), axis=1).mean()
            # XAI term
            d_xai = 0.0
            if xa is not None:
                xs = xa[(xa.dataset == dtype) & (xa.model == model) &
                        (xa.corruption == cname)]
                if len(xs):
                    d_xai = float((1.0 - xs["spearman_clean"]).mean())
            d = W_ACC * d_acc + W_F1 * d_f1 + W_XAI * d_xai
            per_corr.append(d)
            breakdown.append({"dataset": dtype, "model": model,
                              "corruption": cname, "d_acc": d_acc, "d_f1": d_f1,
                              "d_xai": d_xai, "D_c": d})
        prs = float(np.clip(100.0 * (1.0 - sum(per_corr) / len(per_corr)), 0.0, 100.0))
        # plain-accuracy variant for reference
        d_acc_plain = g.groupby("corruption").apply(
            lambda gc: gc.apply(lambda r: relative_drop(c.accuracy, r.accuracy),
                                axis=1).mean(), include_groups=False).mean()
        prs_plain = float(np.clip(100.0 * (1.0 - d_acc_plain), 0.0, 100.0))
        rows.append({"dataset": dtype, "model": model,
                     "PRS": round(prs, 2),
                     "PRS_plain_acc": round(prs_plain, 2),
                     "mean_d_acc": round(float(d_acc_plain), 4),
                     "clean_acc": round(float(c.accuracy), 4),
                     "clean_bacc": round(float(c.balanced_accuracy), 4),
                     "clean_f1": round(float(c.f1), 4),
                     "clean_auc": round(float(c.auc), 4)})
    scores = pd.DataFrame(rows).sort_values(["dataset", "PRS"],
                                            ascending=[True, False])
    scores.to_csv(out_scores, index=False)
    pd.DataFrame(breakdown).to_csv(out_breakdown, index=False)
    print(scores.to_string(index=False))
    print(f"\n[score{tag}] saved -> {out_scores}\n[score{tag}] saved -> {out_breakdown}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_CSV))
    ap.add_argument("--xai", default=str(XAI_CSV))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    compute(results_csv=args.results, xai_csv=Path(args.xai),
            out_scores=RESULTS_DIR / f"robustness_scores{args.tag}.csv",
            out_breakdown=RESULTS_DIR / f"robustness_breakdown{args.tag}.csv",
            tag=args.tag)


from pathlib import Path  # noqa: E402  (used in main only)

if __name__ == "__main__":
    main()
