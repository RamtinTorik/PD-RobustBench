"""Central configuration for the PD drawing robustness benchmark.

Project: Robustness evaluation framework for deep-learning models that detect
Parkinson's disease (PD) from handwriting/drawing images (spiral, wave, meander).
All paths, seeds, model list, corruption definitions and score weights live here.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                # prepared datasets (copied/unzipped here)
CORRUPTED_DIR = DATA_DIR / "corrupted"    # optional on-disk dump of corrupted copies
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"
XAI_DIR = PROJECT_ROOT / "xai_outputs"    # saved Grad-CAM heatmaps
CKPT_DIR = PROJECT_ROOT / "checkpoints"


# Reproducibility / training
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 15          # only the classifier head is trained (light fine-tune)
HEAD_LR = 1e-3
VAL_FRACTION = 0.2        # stratified val split carved out of the training split

# Models (torchvision, ImageNet weights, frozen backbone + light head)
MODELS = ["resnet50", "vgg16", "efficientnet_b0", "mobilenet_v3_large"]

Corruption benchmark  (mirrors ImageNet-C design: 8 families x 5 severities)
# Values are the *parameters* of each severity level (1 = mildest ... 5 = strongest).
CORRUPTION_LEVELS = {
    "jpeg_compression": [90, 70, 50, 30, 10],       # JPEG quality factor (lower = stronger)
    "gaussian_blur":    [3, 5, 7, 9, 11],           # Gaussian kernel size (odd), sigma derived
    "gaussian_noise":   [0.01, 0.03, 0.05, 0.08, 0.12],  # sigma as fraction of 255
    "contrast":         [0.50, 0.70, 0.85, 1.30, 1.50],  # multiplicative factor around mean
    "brightness":       [0.50, 0.70, 0.85, 1.30, 1.50],  # multiplicative factor
    "shadow":           [0.85, 0.70, 0.55, 0.42, 0.30],  # brightness at darkest corner of a diagonal illumination gradient
    "perspective":      [0.02, 0.04, 0.06, 0.08, 0.10],  # corner displacement as fraction of min(h, w)
    "rotation":         [-5, 10, -15, 20, -25],     # degrees, alternating direction, white fill
}

# Robustness score (PRS) weights: accuracy drop, F1 drop, XAI instability
W_ACC, W_F1, W_XAI = 0.5, 0.3, 0.2

# Number of test images per class used for the XAI subset (5 PD + 5 healthy)
XAI_N_PER_CLASS = 5

for _d in (RAW_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, DOCS_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
