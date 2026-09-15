"""Grad-CAM randomization sanity check (review fix 4.7 / medium priority).

Following the model-randomization logic of Adebayo et al. (2018): if the
saliency map is meaningful for the *trained* model, it should decorrelate when
the trainable parameters (head + fine-tuned last stage) are replaced by a
freshly initialized (untrained) model. We compare, on the XAI subset, the
trained model's clean Grad-CAM map with (a) its own map under corruption
(stability, expected to stay correlated) and (b) the untrained model's map
(randomization control, expected to decorrelate).

Output: results/gradcam_sanity.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

from config import CKPT_DIR, RESULTS_DIR, SEED
from data import list_items
from gradcam import GradCAM
from models import PDClassifier, gradcam_target_module, load_model
from run_xai import preprocess_pil, select_subset


def cams_for(model, engine, items, device):
    out = {}
    for path, y, _ in items:
        img = Image.open(path).convert("RGB")
        out[path] = engine(preprocess_pil(img).to(device))
    return out


def main(dtype: str = "pdraw_spiral", seed: int = SEED):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for m in ["resnet50", "vgg16", "efficientnet_b0", "mobilenet_v3_large"]:
        torch.manual_seed(seed)
        trained = load_model(m, CKPT_DIR / f"{dtype}__{m}.pt")
        # untrained control: same architecture, freshly initialized head +
        # last stage (i.e., never trained on this task)
        torch.manual_seed(seed + 1)
        random_model = PDClassifier(m)
        eng_t = GradCAM(trained, gradcam_target_module(trained))
        eng_r = GradCAM(random_model, gradcam_target_module(random_model))
        items = select_subset(list_items(dtype, "test"))
        cams_trained = cams_for(trained, eng_t, items, device)
        cams_random = cams_for(random_model, eng_r, items, device)
        # baseline: CAM magnitude agreement expected to collapse
        for path, y, _ in items:
            a, b = cams_trained[path].ravel(), cams_random[path].ravel()
            if np.std(a) == 0 or np.std(b) == 0:
                # constant map (e.g., all-zero CAM of an untrained model):
                # no rank agreement at all
                rho, note = 0.0, "constant map"
            else:
                rho, note = float(spearmanr(a, b)[0]), ""
            rows.append({"dataset": dtype, "model": m,
                         "image": path.split("/")[-1].split("\\")[-1],
                         "spearman_trained_vs_untrained": round(rho, 4),
                         "note": note})
        eng_t.close(); eng_r.close()
        print(f"[sanity] {m}: mean rho(trained vs untrained) = "
              f"{np.mean([r['spearman_trained_vs_untrained'] for r in rows if r['model'] == m]):.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "gradcam_sanity.csv", index=False)
    print("[sanity] saved ->", RESULTS_DIR / "gradcam_sanity.csv")


if __name__ == "__main__":
    main()
