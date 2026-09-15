"""Aggregate robustness results across the three training seeds (42/43/44).

Reads results/robustness_scores_s{42,43,44}.csv (produced by running
robustness_score.py once per seed) and reports, per (dataset, model):
  - clean metrics: mean +/- std across seeds
  - PRS: mean +/- std across seeds
  - per-seed rank of the model within its dataset and the ranking-stability
    (whether the per-dataset PRS ordering is preserved across seeds)

Outputs: results/seed_aggregate.csv (+ LaTeX rows in results/tables/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import RESULTS_DIR, TABLES_DIR

SEEDS = [42, 43, 44]


def main():
    frames = []
    for s in SEEDS:
        f = RESULTS_DIR / f"robustness_scores_s{s}.csv" if s != 42 else \
            RESULTS_DIR / "robustness_scores.csv"
        df = pd.read_csv(f)
        df["seed"] = s
        frames.append(df)
    all_ = pd.concat(frames, ignore_index=True)

    agg = (all_.groupby(["dataset", "model"])
           .agg(PRS_mean=("PRS", "mean"), PRS_std=("PRS", "std"),
                acc_mean=("clean_acc", "mean"), acc_std=("clean_acc", "std"),
                bacc_mean=("clean_bacc", "mean"), bacc_std=("clean_bacc", "std"),
                auc_mean=("clean_auc", "mean"), auc_std=("clean_auc", "std"),
                f1_mean=("clean_f1", "mean"), f1_std=("clean_f1", "std"))
           .reset_index())
    agg["PRS"] = agg.PRS_mean.round(2).astype(str) + " ± " + agg.PRS_std.round(2).astype(str)
    agg.to_csv(RESULTS_DIR / "seed_aggregate.csv", index=False)

    # ranking stability per dataset across seeds
    print("Per-seed PRS ranking stability (Spearman between seed pairs):")
    for ds, g in all_.groupby("dataset"):
        piv = g.pivot_table(index="model", columns="seed", values="PRS")
        rhos = []
        for i, s1 in enumerate(SEEDS):
            for s2 in SEEDS[i + 1:]:
                if s1 in piv and s2 in piv:
                    rhos.append(spearmanr(piv[s1], piv[s2]).statistic)
        order_42 = piv[42].sort_values(ascending=False).index.tolist()
        print(f"  {ds:16s} rho pairs={[round(r, 2) for r in rhos]} | "
              f"ranking(s42): {order_42}")

    dsmap = {"pdraw_spiral": "PDRAW--spiral", "pdraw_wave": "PDRAW--wave",
             "handpd_spiral": "HandPD--spiral", "handpd_meander": "HandPD--meander"}
    mmap = {"resnet50": "ResNet50", "vgg16": "VGG16",
            "efficientnet_b0": "EffNet-B0", "mobilenet_v3_large": "MobileNetV3"}
    lines = []
    for _, r in agg.iterrows():
        lines.append(
            f"{dsmap[r.dataset]} & {mmap[r.model]} & "
            f"{r.acc_mean:.3f}±{r.acc_std:.3f} & {r.auc_mean:.3f}±{r.auc_std:.3f} & "
            f"{r.PRS_mean:.1f}±{r.PRS_std:.1f} {chr(92)*2}")
    (TABLES_DIR / "table_seed_agg_rows.tex").write_text("\n".join(lines) + "\n")
    print(f"[aggregate] saved -> {RESULTS_DIR / 'seed_aggregate.csv'}")


if __name__ == "__main__":
    main()
