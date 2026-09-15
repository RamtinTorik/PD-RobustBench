"""Train the light classification head on top of frozen ImageNet backbones.

Efficiency trick: backbones are fully frozen and there is no augmentation, so
backbone features for the (clean) training images are extracted exactly once
per (dataset type, model); the head then trains on cached features in seconds.
This is mathematically identical to training with the frozen backbone attached.

Usage:  python train.py            (trains every dataset type x model)
        python train.py --dtype pdraw_spiral --model resnet50
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn

from config import CKPT_DIR, SEED, HEAD_LR, BATCH_SIZE
from data import prepare_all, available_datasets, list_items, stratified_val_split, make_loader
from models import (build_feature_extractor, build_head, PDClassifier,
                    last_stage_module, ckpt_name)
from metrics import best_threshold

FEATURE_EPOCHS = 100
PATIENCE = 15
FT_EPOCHS = 10          # light fine-tuning of the last backbone stage
FT_PATIENCE = 4
FT_BACKBONE_LR = 1e-5
FT_HEAD_LR = 1e-4


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def extract_features(backbone, items, device, batch_size=BATCH_SIZE):
    """Backbone features (N, feat_dim) for a list of (path, label, subj)."""
    loader = make_loader(items, corruption=None, shuffle=False,
                         batch_size=batch_size)
    feats, labels = [], []
    for x, y, _ in loader:
        f = backbone(x.to(device))
        feats.append(f.flatten(1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_head_on_features(feats, labels, val_feats, val_labels,
                           device, seed: int = SEED):
    """Train the small head on cached features; return best head state_dict."""
    torch.manual_seed(seed)
    feat_dim = feats.shape[1]
    head = build_head(feat_dim).to(device)
    # class-balanced loss for imbalanced datasets (pos_weight = n_neg / n_pos)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(head.parameters(), lr=HEAD_LR)

    feats, labels = feats.to(device), labels.float().to(device)
    val_feats, val_labels = val_feats.to(device), val_labels.float().to(device)

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(FEATURE_EPOCHS):
        head.train()
        opt.zero_grad()
        logits = head(feats).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            vp = torch.sigmoid(head(val_feats).squeeze(1)).cpu().numpy()
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(val_labels.cpu().numpy().astype(int), vp)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_state, best_auc, epoch + 1


def _val_probs(model, items, device):
    loader = make_loader(items, shuffle=False, batch_size=32)
    ys, ps = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            ps.append(torch.sigmoid(model(x.to(device))).squeeze(1).cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(ys), np.concatenate(ps)


def _val_auc(model, items, device):
    from sklearn.metrics import roc_auc_score
    y, p = _val_probs(model, items, device)
    try:
        return roc_auc_score(y.astype(int), p)
    except ValueError:
        return 0.5


def train_one(dtype: str, model_name: str, device, seed: int = SEED) -> dict:
    trainval = list_items(dtype, "train")
    if not trainval:
        raise RuntimeError(f"No training images for {dtype}")
    train_items, val_items = stratified_val_split(trainval, seed=seed)
    t0 = time.time()

    # ---- stage 1: train the head on cached frozen-backbone features -------
    backbone, _ = build_feature_extractor(model_name)
    backbone.to(device).eval()
    f_tr, y_tr = extract_features(backbone, train_items, device)
    f_va, y_va = extract_features(backbone, val_items, device)
    head_state, head_val_auc, epochs = train_head_on_features(
        f_tr, y_tr, f_va, y_va, device)
    del backbone

    # ---- stage 2: light fine-tuning of the last backbone stage ------------
    model = PDClassifier(model_name, head_state=head_state).to(device)
    stage = last_stage_module(model)
    for p in stage.parameters():
        p.requires_grad_(True)
    y_arr = np.array([y for _, y, _ in train_items], dtype=float)
    pos_weight = torch.tensor(
        [(len(y_arr) - y_arr.sum()) / max(y_arr.sum(), 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam([
        {"params": stage.parameters(), "lr": FT_BACKBONE_LR},
        {"params": model.head.parameters(), "lr": FT_HEAD_LR},
    ], weight_decay=1e-4)

    best = {"val_auc": -1.0, "state": None, "ep": -1}
    bad = 0
    for ep in range(FT_EPOCHS):
        model.train()
        model.backbone.eval()  # keep backbone BN stats frozen
        for x, y, _ in make_loader(train_items, shuffle=True):
            opt.zero_grad()
            loss = criterion(model(x.to(device)).squeeze(1),
                             y.float().to(device))
            loss.backward()
            opt.step()
        v = _val_auc(model, val_items, device)
        if v > best["val_auc"] + 1e-4:
            best = {"val_auc": v, "ep": ep,
                    "state": {k: t.detach().cpu().clone()
                              for k, t in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= FT_PATIENCE:
                break
    model.load_state_dict(best["state"])

    # ---- decision threshold on the validation split -----------------------
    y_v, p_v = _val_probs(model, val_items, device)
    thr = best_threshold(y_v, p_v)

    # ---- persist only what changes: head + fine-tuned last stage ----------
    stage_state = {k: v.detach().cpu() for k, v in stage.state_dict().items()}
    ckpt_path = CKPT_DIR / ckpt_name(dtype, model_name, seed)
    torch.save({"model": model_name, "dtype": dtype, "seed": seed,
                "head": model.head.state_dict(),
                "last_stage": stage_state,
                "threshold": thr,
                "val_auc_head": head_val_auc, "val_auc_ft": best["val_auc"],
                "ft_epoch": best["ep"], "epochs": epochs},
               ckpt_path)
    dt = time.time() - t0
    return {"dtype": dtype, "model": model_name,
            "val_auc_head": round(head_val_auc, 4),
            "val_auc_ft": round(best["val_auc"], 4),
            "threshold": round(thr, 2), "seconds": round(dt, 1),
            "ckpt": str(ckpt_path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} seed={args.seed}")
    stats = prepare_all()
    print("[train] dataset stats:", json.dumps({f"{k[0]}|{k[1]}|{k[2]}": v
                                                for k, v in stats.items()}))
    print("[train] datasets:", available_datasets())

    dtypes = [args.dtype] if args.dtype else available_datasets()
    models = [args.model] if args.model else None
    models = models or __import__("config").MODELS

    rows = []
    for dtype in dtypes:
        for m in models:
            r = train_one(dtype, m, device, seed=args.seed)
            rows.append(r)
            print(f"[train] {dtype:16s} {m:16s} val_auc_head={r['val_auc_head']:.3f} "
                  f"val_auc_ft={r['val_auc_ft']:.3f} thr={r['threshold']:.2f} "
                  f"({r['seconds']}s)")

    with open(CKPT_DIR / "training_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[train] done ->", CKPT_DIR / "training_summary.csv")


if __name__ == "__main__":
    main()
