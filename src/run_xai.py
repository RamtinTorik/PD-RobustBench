"""XAI experiment (tasks 5.1-5.3): Grad-CAM stability under corruption.

For a fixed subset of test images (XAI_N_PER_CLASS PD + healthy per dataset
type), computes Grad-CAM maps for the clean image and for every corruption
family x severity level, then compares each corrupted map with the clean map:

  - pearson  : Pearson correlation of flattened maps
  - spearman : Spearman rank correlation (primary stability metric)
  - ssim     : structural similarity between the maps
  - iou_top20: IoU of the top-20% most active pixel sets

All maps are saved as .npy under xai_outputs/<dataset>/<model>/<image_stem>/
and overlay PNGs are saved for a small curated selection (paper figure).
Output table: results/xai_similarity.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim_fn

from config import (CORRUPTION_LEVELS, XAI_DIR, RESULTS_DIR, SEED, IMG_SIZE,
                    XAI_N_PER_CLASS, CKPT_DIR, MODELS)
from corruptions import apply_corruption
from data import list_items, DrawDataset
from gradcam import GradCAM
from models import load_model, gradcam_target_module, ckpt_name

OUT_CSV = RESULTS_DIR / "xai_similarity.csv"
FIELDS = ["dataset", "model", "image", "y_true", "corruption", "level",
          "pearson_clean", "spearman_clean", "ssim_clean", "iou_top20_clean"]


def preprocess_pil(img: Image.Image) -> torch.Tensor:
    """Same preprocessing as DrawDataset for a single PIL image -> (1,3,H,W)."""
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


def load_rgb_resized(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize(
        (IMG_SIZE, IMG_SIZE), Image.BICUBIC))


def iou_top(cam_a: np.ndarray, cam_b: np.ndarray, frac: float = 0.2) -> float:
    """IoU between the top-`frac` most active pixels of two maps."""
    n = max(1, int(cam_a.size * frac))
    a = set(np.argsort(cam_a.ravel())[-n:].tolist())
    b = set(np.argsort(cam_b.ravel())[-n:].tolist())
    return len(a & b) / len(a | b)


def select_subset(items, n_per_class: int = XAI_N_PER_CLASS, seed: int = SEED):
    """Deterministic random subset: n PD + n healthy test images."""
    rng = random.Random(seed)
    by = {0: [], 1: []}
    for it in items:
        by[it[1]].append(it)
    sub = []
    for lbl in (0, 1):
        pool = sorted(by[lbl])[:]
        rng.shuffle(pool)
        sub.extend(pool[:n_per_class])
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="pdraw_spiral")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--save_overlays", action="store_true",
                    help="save overlay PNGs for every condition (bigger disk use)")
    args = ap.parse_args()
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [args.model] if args.model else MODELS
    items = select_subset(list_items(args.dtype, "test"))
    print(f"[xai] {args.dtype}: subset of {len(items)} images, models={models}, "
          f"seed={args.seed}")

    write_header = not out_csv.exists()
    for m in models:
        ckpt = CKPT_DIR / ckpt_name(args.dtype, m, args.seed)
        if not ckpt.exists():
            print(f"[xai] SKIP {m}: no checkpoint")
            continue
        model = load_model(m, ckpt)
        cam_engine = GradCAM(model, gradcam_target_module(model))
        for path, ytrue, _subj in items:
            stem = path.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
            out_dir = XAI_DIR / args.dtype / m / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            img_res = load_rgb_resized(path)

            def compute(corruption, level):
                img = Image.open(path).convert("RGB")
                if corruption is not None:
                    arr = apply_corruption(np.array(img), corruption, level)
                    img = Image.fromarray(arr)
                x = preprocess_pil(img).to(device)
                cam = cam_engine(x)
                np.save(out_dir / f"{corruption or 'clean'}_L{level}.npy",
                        cam.astype(np.float32))
                if args.save_overlays:
                    from gradcam import overlay_heatmap
                    Image.fromarray(overlay_heatmap(img_res, cam)).save(
                        out_dir / f"{corruption or 'clean'}_L{level}.png")
                return cam

            cam_clean = compute(None, 0)
            for ctype, levels in CORRUPTION_LEVELS.items():
                for lvl in range(1, len(levels) + 1):
                    cam_c = compute(ctype, lvl)
                    row = {
                        "dataset": args.dtype, "model": m, "image": stem,
                        "y_true": ytrue, "corruption": ctype, "level": lvl,
                        "pearson_clean": round(float(pearsonr(
                            cam_clean.ravel(), cam_c.ravel())[0]), 4),
                        "spearman_clean": round(float(spearmanr(
                            cam_clean.ravel(), cam_c.ravel())[0]), 4),
                        "ssim_clean": round(float(ssim_fn(
                            cam_clean, cam_c, data_range=1.0)), 4),
                        "iou_top20_clean": round(iou_top(cam_clean, cam_c), 4),
                    }
                    with open(out_csv, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=FIELDS)
                        if write_header:
                            w.writeheader(); write_header = False
                        w.writerow(row)
            print(f"  {m:16s} {stem}: done")
        cam_engine.close()
        del model
    print(f"[xai] saved -> {out_csv}")


if __name__ == "__main__":
    main()
