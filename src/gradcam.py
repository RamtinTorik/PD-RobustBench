"""Grad-CAM (Selvaraju et al., ICCV 2017) - minimal hand-written implementation.

Computes class-discriminative localization maps for the PD (positive) logit:
    weights = GAP of dY/dA over spatial dims
    cam     = ReLU(sum_k weights_k * A_k), resized to the input resolution

A is the activation of the backbone's last convolutional stage. Works with the
PDClassifier wrapper (frozen backbone + light head).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_module: nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self._fh = target_module.register_forward_hook(self._save_act)
        self._bh = target_module.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.acts = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """x: (1,3,H,W) preprocessed tensor -> cam (H,W) float in [0,1]."""
        # the backbone is frozen, so the input itself must carry requires_grad
        # for autograd to record the graph through the target conv stage
        x = x.detach().clone().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                     # (1, 1)
        score = logits[0, 0]                       # PD (positive-class) logit
        score.backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)   # (1,C,1,1)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                            align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self):
        self._fh.remove()
        self._bh.remove()


def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] CAM onto the original RGB image with the JET colormap."""
    cam_u8 = np.uint8(255 * cam)
    heat = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    return np.uint8(alpha * heat + (1 - alpha) * img_rgb)
