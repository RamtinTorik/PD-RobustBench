"""Final tables & figures (task 6.2) for the paper.

Produces (from results/*.csv):
  figures/fig_dataset_samples.png      - clean examples (healthy vs PD)
  figures/fig_corruption_grid.png      - one image under all corruptions/levels
  figures/fig_curves_<dtype>.png       - balanced accuracy vs severity per corruption
  figures/fig_robustness_scores.png    - PRS bar chart per model/dataset
  figures/fig_xai_stability.png        - Grad-CAM Spearman corr vs severity
  figures/fig_gradcam_grid.png         - curated Grad-CAM overlays (clean vs corrupt)
  tables/table_clean_performance.csv + .tex
  tables/table_robustness.csv + .tex
  tables/table_xai_stability.csv + .tex
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, XAI_DIR,
                    CORRUPTION_LEVELS, RAW_DIR, CKPT_DIR)
from data import list_items, available_datasets
from models import load_model, gradcam_target_module
from gradcam import GradCAM, overlay_heatmap
from run_xai import preprocess_pil, load_rgb_resized, select_subset

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})

MODEL_LABELS = {"resnet50": "ResNet50", "vgg16": "VGG16",
                "efficientnet_b0": "EfficientNet-B0",
                "mobilenet_v3_large": "MobileNetV3"}
CORR_LABELS = {
    "jpeg_compression": "JPEG compression",
    "gaussian_blur": "Gaussian blur",
    "gaussian_noise": "Gaussian noise",
    "contrast": "Contrast",
    "brightness": "Brightness",
    "shadow": "Shadow",
    "perspective": "Perspective",
    "rotation": "Rotation",
}
DTYPE_LABELS = {"pdraw_spiral": "Parkinson's Drawings - Spiral",
                "pdraw_wave": "Parkinson's Drawings - Wave",
                "handpd_spiral": "HandPD - Spiral",
                "handpd_meander": "HandPD - Meander"}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_dataset_samples():
    from PIL import Image
    items = list_items("pdraw_spiral", "test")
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.2))
    for row, lbl in ((0, 0), (1, 1)):
        sub = [it for it in items if it[1] == lbl][:4]
        cls = "healthy" if lbl == 0 else "parkinson"
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(Image.open(sub[col][0]).convert("L"), cmap="gray")
            ax.set_title(f"{cls} - {sub[col][0].split(chr(92))[-1].split('/')[-1]}",
                         fontsize=7)
            ax.axis("off")
    fig.suptitle("Parkinson's Drawings (spiral): examples of both classes")
    fig.savefig(FIGURES_DIR / "fig_dataset_samples.png")
    plt.close(fig)


def fig_corruption_grid():
    """One PD spiral image under every corruption family x 5 severity levels."""
    from PIL import Image
    from corruptions import apply_corruption
    items = [it for it in list_items("pdraw_spiral", "test") if it[1] == 1]
    img = np.array(Image.open(items[0][0]).convert("RGB"))
    corrs = list(CORRUPTION_LEVELS.keys())
    fig, axes = plt.subplots(len(corrs), 5, figsize=(9, 1.9 * len(corrs)))
    for r, c in enumerate(corrs):
        for lvl in range(1, 6):
            ax = axes[r, lvl - 1]
            ax.imshow(apply_corruption(img, c, lvl))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"level {lvl}", fontsize=8)
        axes[r, 0].text(-0.12, 0.5, CORR_LABELS[c], transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Corruption benchmark applied to a PD spiral (severity 1-5)")
    fig.savefig(FIGURES_DIR / "fig_corruption_grid.png")
    plt.close(fig)


def fig_curves(dtype: str):
    df = pd.read_csv(RESULTS_DIR / "robustness_results.csv")
    d = df[df.dataset == dtype]
    if not len(d):
        return
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    for ax, (corr, levels) in zip(axes.ravel(), CORRUPTION_LEVELS.items()):
        for m in d.model.unique():
            sub = d[(d.model == m) & (d.corruption == corr)].sort_values("level")
            clean_val = d[(d.model == m) & (d.condition_type == "clean")][
                "balanced_accuracy"]
            base = float(clean_val.iloc[0]) if len(clean_val) else np.nan
            ax.plot(sub.level, sub.balanced_accuracy, "o-", ms=3,
                    label=MODEL_LABELS.get(m, m))
            ax.axhline(base, ls=":", lw=0.8, color="gray")
        ax.set_title(CORR_LABELS[corr], fontsize=9)
        ax.set_xlabel("severity level")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("balanced accuracy")
    axes[1, 0].set_ylabel("balanced accuracy")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Degradation under corruptions - {DTYPE_LABELS.get(dtype, dtype)}"
                 " (dotted = clean baseline)")
    fig.savefig(FIGURES_DIR / f"fig_curves_{dtype}.png")
    plt.close(fig)


def fig_robustness_scores():
    df = pd.read_csv(TABLES_DIR / "table_robustness.csv")
    dsets = df.dataset.unique()
    fig, axes = plt.subplots(1, len(dsets), figsize=(3.4 * len(dsets), 3.4),
                             sharey=True)
    if len(dsets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, dsets):
        sub = df[df.dataset == ds]
        ax.bar([MODEL_LABELS.get(m, m) for m in sub.model], sub.PRS,
               color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(sub)])
        ax.set_title(DTYPE_LABELS.get(ds, ds), fontsize=9)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=20, labelsize=7)
        for i, v in enumerate(sub.PRS):
            ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Parkinson Robustness Score (PRS)")
    fig.savefig(FIGURES_DIR / "fig_robustness_scores.png")
    plt.close(fig)


def fig_xai_stability(dtype: str):
    path = RESULTS_DIR / "xai_similarity.csv"
    df = pd.read_csv(path)
    d = df[df.dataset == dtype]
    if not len(d):
        return
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharey=True)
    for ax, corr in zip(axes.ravel(), CORRUPTION_LEVELS.keys()):
        for m in d.model.unique():
            sub = d[(d.model == m) & (d.corruption == corr)].groupby("level")[
                "spearman_clean"].mean()
            ax.plot(sub.index, sub.values, "o-", ms=3, label=MODEL_LABELS.get(m, m))
        ax.set_title(CORR_LABELS[corr], fontsize=9)
        ax.set_xlabel("severity level")
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.1, 1.05)
    axes[0, 0].set_ylabel("Spearman corr. vs clean CAM")
    axes[1, 0].set_ylabel("Spearman corr. vs clean CAM")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Grad-CAM stability under corruption - {DTYPE_LABELS.get(dtype, dtype)}")
    fig.savefig(FIGURES_DIR / f"fig_xai_stability_{dtype}.png")
    plt.close(fig)


def fig_gradcam_grid(dtype: str = "pdraw_spiral", model: str = "resnet50"):
    """Curated panel: clean vs three corruptions at high severity, for one
    PD and one healthy image."""
    items = select_subset(list_items(dtype, "test"))
    model_obj = load_model(model, CKPT_DIR / f"{dtype}__{model}.pt")
    engine = GradCAM(model_obj, gradcam_target_module(model_obj))
    picks = [next(it for it in items if it[1] == 1),
             next(it for it in items if it[1] == 0)]
    show = [("clean", 0), ("gaussian_noise", 5), ("jpeg_compression", 5),
            ("gaussian_blur", 5)]
    fig, axes = plt.subplots(len(picks), len(show), figsize=(8, 4.4))
    from PIL import Image
    for r, (path, y, _) in enumerate(picks):
        img_res = load_rgb_resized(path)
        for c, (corr, lvl) in enumerate(show):
            img = Image.open(path).convert("RGB")
            if corr != "clean":
                from corruptions import apply_corruption
                img = Image.fromarray(apply_corruption(np.array(img), corr, lvl))
            cam = engine(preprocess_pil(img))
            axes[r, c].imshow(overlay_heatmap(img_res, cam))
            title = CORR_LABELS.get(corr, corr) + (f" L{lvl}" if corr != "clean" else "")
            axes[r, c].set_title(f"{'PD' if y else 'Healthy'} | {title}", fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle(f"Grad-CAM ({MODEL_LABELS[model]}, {DTYPE_LABELS.get(dtype, dtype)}): "
                 "PD-class evidence, clean vs corrupted (level 5)")
    fig.savefig(FIGURES_DIR / f"fig_gradcam_{dtype}_{model}.png")
    plt.close(fig)
    engine.close()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table_clean_performance():
    df = pd.read_csv(RESULTS_DIR / "robustness_results.csv")
    clean = df[df.condition_type == "clean"].copy()
    cols = ["dataset", "model", "accuracy", "balanced_accuracy", "precision",
            "sensitivity", "specificity", "f1", "auc"]
    t = clean[cols].copy()
    for c in cols[2:]:
        t[c] = t[c].round(4)
    t = t.sort_values(["dataset", "model"])
    t.to_csv(TABLES_DIR / "table_clean_performance.csv", index=False)
    latex = t.to_latex(index=False, float_format="%.3f",
                       column_format="llccccccc",
                       header=["Dataset", "Model", "Acc.", "Bal. Acc.",
                               "Prec.", "Sens.", "Spec.", "F1", "AUC"])
    (TABLES_DIR / "table_clean_performance.tex").write_text(latex)
    return t


def table_robustness():
    df = pd.read_csv(RESULTS_DIR / "robustness_breakdown.csv")
    piv = df.pivot_table(index=["dataset", "model"], columns="corruption",
                         values="D_c")
    scores = pd.read_csv(RESULTS_DIR / "robustness_scores.csv")[
        ["dataset", "model", "PRS"]].set_index(["dataset", "model"])
    t = piv.join(scores).reset_index()
    cols = ["dataset", "model"] + list(CORRUPTION_LEVELS.keys()) + ["PRS"]
    t = t[cols].round(4).sort_values(["dataset", "PRS"], ascending=[True, False])
    t.to_csv(TABLES_DIR / "table_robustness.csv", index=False)
    latex = t.to_latex(index=False, float_format="%.3f", na_rep="-",
                       column_format="llccccccccc",
                       header=["Dataset", "Model"] + [CORR_LABELS[c] for c in
                                                      CORRUPTION_LEVELS] + ["PRS"])
    (TABLES_DIR / "table_robustness.tex").write_text(latex)
    return t


def table_xai_stability():
    df = pd.read_csv(RESULTS_DIR / "xai_similarity.csv")
    t = (df.groupby(["dataset", "model", "corruption"])["spearman_clean"]
         .mean().unstack("corruption").round(3).reset_index())
    cols = ["dataset", "model"] + [c for c in CORRUPTION_LEVELS if c in t.columns]
    t = t[cols]
    t.to_csv(TABLES_DIR / "table_xai_stability.csv", index=False)
    latex = t.to_latex(index=False, na_rep="-", column_format="llcccccccc",
                       header=["Dataset", "Model"] + [CORR_LABELS[c] for c in
                                                      CORRUPTION_LEVELS])
    (TABLES_DIR / "table_xai_stability.tex").write_text(latex)
    return t


if __name__ == "__main__":
    import torch
    fig_dataset_samples()
    fig_corruption_grid()
    for ds in available_datasets():
        fig_curves(ds)
    if (RESULTS_DIR / "robustness_scores.csv").exists():
        table_robustness()
        fig_robustness_scores()
    table_clean_performance()
    if (RESULTS_DIR / "xai_similarity.csv").exists():
        table_xai_stability()
        fig_xai_stability("pdraw_spiral")
        try:
            fig_gradcam_grid("pdraw_spiral", "resnet50")
        except Exception as e:
            print("gradcam figure skipped:", e)
    print("figures/tables done ->", FIGURES_DIR, TABLES_DIR)
