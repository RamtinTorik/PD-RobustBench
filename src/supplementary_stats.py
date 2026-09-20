"""Supplementary statistics (review fixes):
1. Bootstrap 95% CIs for clean accuracy / balanced accuracy / AUC
   (resampling per-image predictions, 2000 resamples, seed 42).
2. Trainable vs total parameter counts per backbone.
3. Per-model decision thresholds (validation-selected) as a table.

Outputs: results/bootstrap_ci.csv, results/model_params.csv,
         results/thresholds.csv (+ LaTeX row files in results/tables/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from config import CKPT_DIR, RESULTS_DIR, TABLES_DIR, SEED
from data import available_datasets
from models import load_model, load_threshold
from inference import predict
from data import list_items

RNG = np.random.default_rng(SEED)
N_BOOT = 2000


def bootstrap_ci(y, p, thr):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n = len(y)
    accs, baccs, aucs = [], [], []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        accs.append(accuracy_score(yb, (pb >= thr).astype(int)))
        baccs.append(balanced_accuracy_score(yb, (pb >= thr).astype(int)))
        try:
            aucs.append(roc_auc_score(yb, pb))
        except ValueError:
            aucs.append(np.nan)
    def ci(vals):
        vals = np.asarray(vals, dtype=float)
        return (np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5))
    point = dict(
        accuracy=accuracy_score(y, (p >= thr).astype(int)),
        balanced_accuracy=balanced_accuracy_score(y, (p >= thr).astype(int)),
        auc=roc_auc_score(y, p),
    )
    lo_acc, hi_acc = ci(accs)
    lo_b, hi_b = ci(baccs)
    lo_a, hi_a = ci(aucs)
    return {
        "accuracy": (point["accuracy"], lo_acc, hi_acc),
        "balanced_accuracy": (point["balanced_accuracy"], lo_b, hi_b),
        "auc": (point["auc"], lo_a, hi_a),
    }


def main():
    # 1. bootstrap CIs on clean predictions
    preds = pd.read_csv(RESULTS_DIR / "predictions.csv")
    clean = preds[preds.corruption == "clean"]
    rows = []
    for (dtype, model), g in clean.groupby(["dataset", "model"]):
        thr = load_threshold(CKPT_DIR / f"{dtype}__{model}.pt")
        cis = bootstrap_ci(g.y_true, g.y_prob, thr)
        for metric, (pt, lo, hi) in cis.items():
            rows.append({"dataset": dtype, "model": model, "metric": metric,
                         "point": round(pt, 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4)})
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)

    dsmap = {"pdraw_spiral": "PDRAW--spiral", "pdraw_wave": "PDRAW--wave",
             "handpd_spiral": "HandPD--spiral", "handpd_meander": "HandPD--meander"}
    mmap = {"resnet50": "ResNet50", "vgg16": "VGG16",
            "efficientnet_b0": "EffNet-B0", "mobilenet_v3_large": "MobileNetV3"}
    lines = []
    for _, r in ci_df[ci_df.metric.isin(["accuracy", "auc"])].iterrows():
        lines.append(f"{dsmap[r.dataset]} & {mmap[r.model]} & "
                     f"{'Accuracy' if r.metric == 'accuracy' else 'ROC-AUC'} & "
                     f"{r.point:.3f} & [{r.ci_low:.3f}, {r.ci_high:.3f}] {chr(92)*2}")
    (TABLES_DIR / "table_ci_rows.tex").write_text("\n".join(lines) + "\n")

    # 2. parameter counts
    from models import PDClassifier, last_stage_module
    prows = []
    for m in ["resnet50", "vgg16", "efficientnet_b0", "mobilenet_v3_large"]:
        model = PDClassifier(m)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in last_stage_module(model).parameters()) + \
            sum(p.numel() for p in model.head.parameters())
        prows.append({"model": m, "params_total_m": round(total / 1e6, 1),
                      "params_trainable_m": round(trainable / 1e6, 2)})
    pd.DataFrame(prows).to_csv(RESULTS_DIR / "model_params.csv", index=False)
    plines = [f"{mmap[r['model']]} & {r.params_total_m:.1f}M & "
              f"{r.params_trainable_m:.2f}M {chr(92)*2}"
              for _, r in pd.DataFrame(prows).iterrows()]
    (TABLES_DIR / "table_params_rows.tex").write_text("\n".join(plines) + "\n")

    # 3. thresholds
    trows = []
    for dtype in available_datasets():
        for m in ["resnet50", "vgg16", "efficientnet_b0", "mobilenet_v3_large"]:
            thr = load_threshold(CKPT_DIR / f"{dtype}__{m}.pt")
            trows.append({"dataset": dtype, "model": m, "threshold": thr})
    pd.DataFrame(trows).to_csv(RESULTS_DIR / "thresholds.csv", index=False)
    tlines = [f"{dsmap[r['dataset']]} & {mmap[r['model']]} & {r.threshold:.2f} {chr(92)*2}"
              for _, r in pd.DataFrame(trows).iterrows()]
    (TABLES_DIR / "table_thresholds_rows.tex").write_text("\n".join(tlines) + "\n")

    print(ci_df.to_string(index=False))
    print(pd.DataFrame(prows).to_string(index=False))
    print("[stats] saved bootstrap_ci.csv, model_params.csv, thresholds.csv")


if __name__ == "__main__":
    main()
