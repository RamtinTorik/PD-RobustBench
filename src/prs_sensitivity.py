"""PRS weight-sensitivity analysis (review fix: ablation over weights).

Recomputes PRS under three weightings from the released per-family
decomposition (results/robustness_breakdown.csv):
  W1 (main)  = (0.50, 0.30, 0.20)  accuracy / F1 / XAI
  W2 (perf)  = (0.70, 0.30, 0.00)  performance-only
  W3 (equal) = (1/3, 1/3, 1/3)
PRS is clipped to [0, 100]. Reports per-configuration scores and the Spearman
rank correlation of the model ranking per dataset between W1 and the variants.

Outputs: results/prs_sensitivity.csv (+ LaTeX rows in results/tables/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import RESULTS_DIR, TABLES_DIR

WEIGHTS = {
    "W1_main": (0.50, 0.30, 0.20),
    "W2_perf": (0.70, 0.30, 0.00),
    "W3_equal": (1 / 3, 1 / 3, 1 / 3),
}


def prs_from_breakdown(bd: pd.DataFrame, w) -> pd.Series:
    s = bd.groupby(["dataset", "model"]).apply(
        lambda g: float(np.clip(100 * (1 - (w[0] * g.d_acc + w[1] * g.d_f1 +
                                           w[2] * g.d_xai).mean()), 0, 100)),
        include_groups=False)
    return s


def main():
    bd = pd.read_csv(RESULTS_DIR / "robustness_breakdown.csv")
    scores = {name: prs_from_breakdown(bd, w) for name, w in WEIGHTS.items()}
    out = pd.DataFrame(scores).reset_index()
    out.to_csv(RESULTS_DIR / "prs_sensitivity.csv", index=False)

    print(out.round(2).to_string(index=False))
    print("\nRank agreement (Spearman of model ranking per dataset, vs W1):")
    for name in ("W2_perf", "W3_equal"):
        rhos = []
        for ds, g in out.groupby("dataset"):
            g = g.sort_values("model")
            rho = spearmanr(g["W1_main"], g[name]).statistic
            rhos.append(rho)
        print(f"  {name}: mean rho = {np.mean(rhos):.2f} (per-dataset {np.round(rhos, 2)})")

    dsmap = {"pdraw_spiral": "PDRAW--spiral", "pdraw_wave": "PDRAW--wave",
             "handpd_spiral": "HandPD--spiral", "handpd_meander": "HandPD--meander"}
    mmap = {"resnet50": "ResNet50", "vgg16": "VGG16",
            "efficientnet_b0": "EffNet-B0", "mobilenet_v3_large": "MobileNetV3"}
    lines = []
    for _, r in out.iterrows():
        lines.append(f"{dsmap[r.dataset]} & {mmap[r.model]} & "
                     f"{r.W1_main:.1f} & {r.W2_perf:.1f} & {r.W3_equal:.1f} {chr(92)*2}")
    (TABLES_DIR / "table_prs_sens_rows.tex").write_text("\n".join(lines) + "\n")
    print(f"[sensitivity] saved -> {RESULTS_DIR / 'prs_sensitivity.csv'}")


if __name__ == "__main__":
    main()
