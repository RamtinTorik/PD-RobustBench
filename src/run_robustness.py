"""Main robustness evaluation loop (tasks 4.1-4.3).

For every dataset type x model, runs inference on the clean test set and on
every corruption family x severity level (corrupted on the fly, in memory),
computes all metrics and appends one row per (dtype, model, condition, level)
to results/robustness_results.csv. Per-image predictions are saved to
results/predictions.csv for later XAI-subset selection and auditing.

Usage:  python run_robustness.py [--dtype pdraw_spiral] [--model resnet50]
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import torch

from config import RESULTS_DIR, CORRUPTION_LEVELS, SEED, CKPT_DIR, MODELS
from data import available_datasets, list_items
from inference import predict
from metrics import compute_metrics
from models import load_model, load_threshold, ckpt_name

RESULTS_CSV = RESULTS_DIR / "robustness_results.csv"
PRED_CSV = RESULTS_DIR / "predictions.csv"

METRIC_FIELDS = ["n", "accuracy", "balanced_accuracy", "precision",
                 "sensitivity", "specificity", "f1", "auc",
                 "tp", "fp", "tn", "fn"]

RESULT_FIELDS = ["dataset", "model", "condition_type", "corruption", "level",
                 "param"] + METRIC_FIELDS
PRED_FIELDS = ["dataset", "model", "corruption", "level", "path",
               "y_true", "y_prob", "y_pred"]


def conditions():
    """Yield (corruption, level, param): clean + all corruption x level combos."""
    yield "clean", 0, ""
    for ctype, levels in CORRUPTION_LEVELS.items():
        for lvl, param in enumerate(levels, start=1):
            yield ctype, lvl, param


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(RESULTS_CSV),
                    help="output CSV (use a per-process file for parallel runs)")
    ap.add_argument("--save_predictions", action="store_true", default=True)
    args = ap.parse_args()
    results_csv = Path(args.out)
    pred_csv = results_csv.with_name("predictions_" + results_csv.stem.replace("robustness_", "") + ".csv") \
        if results_csv.name != RESULTS_CSV.name else PRED_CSV
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[robustness] device={device}")

    dtypes = [args.dtype] if args.dtype else available_datasets()
    models = [args.model] if args.model else MODELS

    write_header = not results_csv.exists()
    pred_header = not pred_csv.exists()

    t_start = time.time()
    for dtype in dtypes:
        test_items = list_items(dtype, "test")
        for m in models:
            ckpt = CKPT_DIR / ckpt_name(dtype, m, args.seed)
            if not ckpt.exists():
                print(f"[robustness] SKIP {dtype}/{m}: no checkpoint {ckpt}")
                continue
            model = load_model(m, ckpt)
            thr = load_threshold(ckpt)
            n_conds = 1 + sum(len(v) for v in CORRUPTION_LEVELS.values())
            print(f"[robustness] {dtype} x {m} (seed {args.seed}): "
                  f"{len(test_items)} images, {n_conds} conditions, thr={thr:.2f}")
            for corruption, level, param in conditions():
                corr = None if corruption == "clean" else (corruption, level)
                y, p, paths = predict(model, test_items, device, corruption=corr)
                met = compute_metrics(y, p, threshold=thr)
                row = {"dataset": dtype, "model": m,
                       "condition_type": "clean" if corruption == "clean" else "corrupted",
                       "corruption": corruption, "level": level, "param": param,
                       **met}
                with open(results_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
                    if write_header:
                        w.writeheader(); write_header = False
                    w.writerow(row)
                if args.save_predictions:
                    ypred = (p >= thr).astype(int)
                    with open(pred_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
                        if pred_header:
                            w.writeheader(); pred_header = False
                        for yi, pi, pi_cls, path in zip(y, p, ypred, paths):
                            w.writerow({"dataset": dtype, "model": m,
                                        "corruption": corruption, "level": level,
                                        "path": path, "y_true": int(yi),
                                        "y_prob": f"{float(pi):.6f}",
                                        "y_pred": int(pi_cls)})
                print(f"  {corruption:18s} L{level} acc={met['accuracy']:.3f} "
                      f"sens={met['sensitivity']:.3f} spec={met['specificity']:.3f}")
            del model
    print(f"[robustness] done in {(time.time()-t_start)/60:.1f} min -> {results_csv}")


if __name__ == "__main__":
    main()
