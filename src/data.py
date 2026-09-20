"""Dataset preparation and loading for the PD drawing robustness benchmark.

Two sources are supported:

1. PDRAW — "Parkinson's Drawings" (Zham et al. 2017; Kaggle release by
   K. Scott Mader). 256x256 PNGs, official train/test split, classes
   healthy/parkinson, data types spiral and wave.

2. HandPD (Pereira et al. 2016) — scanned micrograph images (~700x700 JPGs),
   classes Control(=healthy)/Patients(=PD), data types spiral and meander,
   92 subjects x 4 images. No official split -> we create a *subject-wise*
   stratified 70/30 split (no subject appears in both splits).

Directory layout produced under data/raw/:
    data/raw/<dataset>_<dtype>/<split>/<class>/*.{png,jpg}
e.g. data/raw/pdraw_spiral/train/healthy/V01HE02.png
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import (RAW_DIR, DATA_DIR, SEED, IMG_SIZE, BATCH_SIZE,
                    VAL_FRACTION)

# healthy = 0 (negative), parkinson = 1 (positive)
LABELS = {"healthy": 0, "parkinson": 1, "control": 0, "patients": 1}
_CANON = {"healthy": "healthy", "control": "healthy",
          "parkinson": "parkinson", "patients": "parkinson"}

KAGGLE_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / "kmader" / \
    "parkinsons-drawings" / "versions" / "1"
HANDPD_TMP = DATA_DIR / "handpd_tmp"

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


# Preparation
def prepare_pdraw() -> dict:
    """Copy the Kaggle Parkinson's Drawings PNGs into data/raw/pdraw_<dtype>/."""
    out = {}
    for dtype in ("spiral", "wave"):
        src_root = KAGGLE_CACHE / dtype
        if not src_root.exists():
            raise FileNotFoundError(f"Kaggle dataset not found at {src_root}")
        for split in ("training", "testing"):
            for cls in ("healthy", "parkinson"):
                src = src_root / split / cls
                dst = RAW_DIR / f"pdraw_{dtype}" / \
                    ("train" if split == "training" else "test") / cls
                dst.mkdir(parents=True, exist_ok=True)
                n = 0
                for f in src.iterdir():
                    if f.suffix.lower() in IMAGE_EXTS:
                        target = dst / f.name
                        if not target.exists():
                            shutil.copy2(f, target)
                        n += 1
                out[(dtype, split, cls)] = n
    return out


def prepare_handpd() -> dict:
    """Copy HandPD JPGs into a subject-wise stratified 70/30 train/test split."""
    rng = random.Random(SEED)
    out = {}
    for dtype, prefix in (("spiral", "Spiral"), ("meander", "Meander")):
        # collect subjects per class: {subject_id: [files]}
        subjects = {"healthy": {}, "parkinson": {}}
        for cls, folder in (("healthy", f"{prefix}Control"),
                            ("parkinson", f"{prefix}Patients")):
            src = HANDPD_TMP / folder
            for f in sorted(src.glob("*.jpg")):
                subj = f.stem.split("-")[0]
                subjects[cls].setdefault(subj, []).append(f)
        split_map = {}
        for cls, subs in subjects.items():
            sub_ids = sorted(subs)
            rng.shuffle(sub_ids)
            n_test = max(3, round(len(sub_ids) * 0.30))
            test_ids = set(sub_ids[:n_test])
            for sid in sub_ids:
                split_map[sid] = ("test" if sid in test_ids else "train", cls)
        # copy files
        for sid, (split, cls) in split_map.items():
            for f in subjects[cls][sid]:
                dst = RAW_DIR / f"handpd_{dtype}" / split / _CANON[cls]
                dst.mkdir(parents=True, exist_ok=True)
                target = dst / f.name
                if not target.exists():
                    shutil.copy2(f, target)
        for split in ("train", "test"):
            for cls in ("healthy", "parkinson"):
                d = RAW_DIR / f"handpd_{dtype}" / split / cls
                out[(dtype, split, cls)] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return out


def prepare_all() -> dict:
    stats = {}
    if KAGGLE_CACHE.exists():
        stats.update(prepare_pdraw())
    if HANDPD_TMP.exists():
        stats.update(prepare_handpd())
    return stats


# Listing / loading
 available_datasets() -> list:
    """Return keys of prepared dataset types, e.g. ['pdraw_spiral', ...]."""
    keys = []
    for d in sorted(RAW_DIR.iterdir()):
        if d.is_dir() and (d / "test").exists():
            keys.append(d.name)
    return keys


def list_items(dataset_key: str, split: str) -> list:
    """Return sorted [(path, label, subject_id)] for one split."""
    items = []
    root = RAW_DIR / dataset_key / split
    for cls in ("healthy", "parkinson"):
        label = LABELS[cls]
        d = root / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                subj = f.stem.split("-")[0]
                items.append((str(f), label, subj))
    return items


def stratified_val_split(items: list, val_fraction: float = VAL_FRACTION,
                         seed: int = SEED) -> tuple:
    """Split a list of items into (train, val) stratified by label."""
    rng = random.Random(seed)
    by_label = {}
    for it in items:
        by_label.setdefault(it[1], []).append(it)
    train, val = [], []
    for _, group in sorted(by_label.items()):
        group = group[:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction))
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val


class DrawDataset(Dataset):
    """Torch dataset over (path, label) items with optional on-the-fly corruption.

    corruption=None            -> clean images
    corruption=(name, level)   -> apply the named corruption at severity level
    """

    def __init__(self, items, corruption=None, img_size: int = IMG_SIZE):
        self.items = items
        self.corruption = corruption  # None or (name, level)
        from corruptions import apply_corruption
        self._apply = apply_corruption
        import torchvision.transforms as T
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, subj = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.corruption is not None:
            arr = np.array(img)
            arr = self._apply(arr, self.corruption[0], self.corruption[1])
            img = Image.fromarray(arr)
        return self.tf(img), label, path


def make_loader(items, corruption=None, shuffle=False, batch_size=BATCH_SIZE,
                seed: int = SEED):
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(DrawDataset(items, corruption=corruption),
                      batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=0, pin_memory=False)


def dataset_summary() -> str:
    lines = []
    for key in available_datasets():
        for split in ("train", "test"):
            items = list_items(key, split)
            n_pd = sum(1 for it in items if it[1] == 1)
            lines.append(f"{key:18s} {split:5s}: {len(items):4d} images "
                         f"({n_pd} parkinson / {len(items) - n_pd} healthy)")
    return "\n".join(lines)
