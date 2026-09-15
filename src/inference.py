"""Inference helpers: run a trained model over a dataset and collect predictions.

Usage (task 4.1 - clean baseline predictions):
    python inference.py --dtype pdraw_spiral --model resnet50
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from config import CKPT_DIR
from data import list_items, make_loader
from models import load_model, load_threshold
from metrics import compute_metrics, format_metrics


@torch.no_grad()
def predict(model, items, device, corruption=None, batch_size=32):
    """Return (y_true, y_prob, paths) for items, optionally under corruption."""
    loader = make_loader(items, corruption=corruption, shuffle=False,
                         batch_size=batch_size)
    model.to(device).eval()
    ys, ps, paths = [], [], []
    for x, y, p in loader:
        logits = model(x.to(device)).squeeze(1)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(np.asarray(y))
        ps.append(prob)
        paths.extend(p)
    return np.concatenate(ys), np.concatenate(ps), paths


def run_clean_inference(dtype: str, model_name: str, device="cpu"):
    ckpt = CKPT_DIR / f"{dtype}__{model_name}.pt"
    model = load_model(model_name, ckpt)
    thr = load_threshold(ckpt)
    test_items = list_items(dtype, "test")
    y, p, _ = predict(model, test_items, device, corruption=None)
    m = compute_metrics(y, p, threshold=thr)
    print(f"[clean] {dtype} / {model_name}: {format_metrics(m)}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="pdraw_spiral")
    ap.add_argument("--model", default="resnet50")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_clean_inference(args.dtype, args.model, dev)
