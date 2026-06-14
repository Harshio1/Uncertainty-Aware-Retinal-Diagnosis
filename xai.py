"""
xai.py — Grad-CAM, saliency, faithfulness, stability, and localisation utilities.

FIXES vs original:
  - Added `auc()` helper (was referenced but never defined → NameError).
  - All `torch.amp.autocast` calls are now device-conditional via
    contextlib.nullcontext() so the module runs on CPU (HF Spaces default).
  - Minor: import cleanup.
"""

import gc
import math
import io
import base64
from contextlib import nullcontext

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from normative import NORMAL_MEAN as mean_normal
from normative import NORMAL_STD as std_normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])


# ── Helper ─────────────────────────────────────────────────────────────────

def _autocast_ctx():
    """Return an autocast context manager appropriate for the current device."""
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda")
    return nullcontext()


def auc(x, y) -> float:
    """Trapezoidal AUC (area under curve). Replacement for missing import."""
    return float(np.trapz(y, x))


# ── Grad-CAM ───────────────────────────────────────────────────────────────

class NativeGradCAM:
    def __init__(self, model):
        self.model  = model
        self.grads  = None
        self.acts   = None
        self.handles = []
        tgt = model.vis_backbone.blocks[-1]
        self.handles.append(
            tgt.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        )
        self.handles.append(
            tgt.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "grads", go[0])
            )
        )

    def generate(self, iv, ig, target, use_unet=True):
        self.model.zero_grad()
        lg, _, attn = self.model(iv, ig, use_unet=use_unet)
        lg[0, target].backward(retain_graph=True)
        w   = torch.mean(self.grads, dim=[2, 3], keepdim=True)
        cam = F.relu(torch.sum(w * self.acts, dim=1, keepdim=True))
        cam = F.interpolate(cam, (224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam_norm, attn

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ── Saliency ───────────────────────────────────────────────────────────────

def compute_saliency(model, vis, geo, label):
    """
    Returns: gcam, cafm_norm, combined, vessel_map, img_display

    Combined saliency: S = 0.6 * L2norm(Grad-CAM) + 0.4 * L2norm(CAFM)
    Visual-stream priority (higher spatial resolution).
    """
    vis_g = vis.clone().to(device).requires_grad_(True)
    geo_d = geo.to(device)

    gc_obj = NativeGradCAM(model)
    gcam, attn = gc_obj.generate(vis_g, geo_d, label, use_unet=True)
    gc_obj.remove_hooks()

    # CAFM attention: [B, num_heads, 1, 49] → mean over heads → [49] → [7,7]
    cafm_raw = attn[0].mean(dim=0).view(7, 7).cpu().detach().numpy()
    cafm = cv2.resize(cafm_raw, (224, 224), interpolation=cv2.INTER_CUBIC)
    cafm = np.clip(cafm, 0, None)

    def l2_normalize(m):
        m_min, m_max = m.min(), m.max()
        if m_max - m_min < 1e-8:
            return np.zeros_like(m)
        return (m - m_min) / (m_max - m_min)

    gcam_norm = l2_normalize(gcam)
    cafm_norm = l2_normalize(cafm)

    alpha    = 0.6
    combined = alpha * gcam_norm + (1 - alpha) * cafm_norm
    combined = l2_normalize(combined)

    # U-Net vessel probability map
    with torch.no_grad(), _autocast_ctx():
        vmap = model.unet(geo_d)
    geo_224 = (
        F.interpolate(vmap, (224, 224), mode="bilinear", align_corners=False)
        .squeeze()
        .cpu()
        .float()
        .numpy()
    )

    # Denormalize input for display
    img_np = vis.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * IMG_STD + IMG_MEAN, 0, 1)

    return gcam, cafm_norm, combined, geo_224, img_np


# ── Faithfulness ───────────────────────────────────────────────────────────

def compute_faithfulness(model, vis, geo, saliency, label, steps=10):
    """
    Dual-stream faithfulness: deletion AUC + insertion AUC.
    Lower deletion AUC and higher insertion AUC = more faithful saliency.
    """
    vis_d, geo_d = vis.to(device), geo.to(device)

    img_np = np.clip(
        vis.squeeze().cpu().numpy().transpose(1, 2, 0) * IMG_STD + IMG_MEAN, 0, 1
    ).astype(np.float32)
    blurred_vis = cv2.GaussianBlur(img_np, (51, 51), 0)

    geo_np      = geo.squeeze().cpu().numpy()          # [512, 512]
    blurred_geo = cv2.GaussianBlur(geo_np, (51, 51), 0)

    sort_idx_vis  = np.argsort(saliency.flatten())[::-1]
    step_size_vis = max(1, len(sort_idx_vis) // steps)

    saliency_geo  = cv2.resize(
        saliency, (geo_np.shape[1], geo_np.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    sort_idx_geo  = np.argsort(saliency_geo.flatten())[::-1]
    step_size_geo = max(1, len(sort_idx_geo) // steps)

    def vis_to_tensor(arr):
        t = (arr - IMG_MEAN) / IMG_STD
        return (
            torch.tensor(t.transpose(2, 0, 1), dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

    def geo_to_tensor(arr):
        return (
            torch.tensor(arr, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.eval()

    del_p, ins_p = [], []

    for s in range(steps + 1):
        px_vis = sort_idx_vis[: s * step_size_vis]
        px_geo = sort_idx_geo[: s * step_size_geo]

        d_vis = img_np.copy().reshape(-1, 3)
        d_vis[px_vis] = blurred_vis.reshape(-1, 3)[px_vis]
        d_geo = geo_np.copy().flatten()
        d_geo[px_geo] = blurred_geo.flatten()[px_geo]
        d_geo = d_geo.reshape(geo_np.shape)

        i_vis = blurred_vis.copy().reshape(-1, 3)
        i_vis[px_vis] = img_np.reshape(-1, 3)[px_vis]
        i_geo = blurred_geo.copy().flatten()
        i_geo[px_geo] = geo_np.flatten()[px_geo]
        i_geo = i_geo.reshape(geo_np.shape)

        with torch.no_grad(), _autocast_ctx():
            d_out, _, _ = model(
                vis_to_tensor(d_vis.reshape(224, 224, 3)),
                geo_to_tensor(d_geo),
                use_unet=True,
            )
            i_out, _, _ = model(
                vis_to_tensor(i_vis.reshape(224, 224, 3)),
                geo_to_tensor(i_geo),
                use_unet=True,
            )

        del_p.append(F.softmax(d_out.float(), 1)[0, label].item())
        ins_p.append(F.softmax(i_out.float(), 1)[0, label].item())

    fracs = np.linspace(0, 1, steps + 1)
    return auc(fracs, del_p), auc(fracs, ins_p), fracs, del_p, ins_p


# ── Stability ──────────────────────────────────────────────────────────────

def compute_stability(model, vis, geo, label, T=10):
    """MC-Dropout Grad-CAM stability with aggressive memory cleanup."""
    geo_d = geo.to(device)
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

    maps = []
    for _ in range(T):
        vt   = vis.clone().to(device).requires_grad_(True)
        gc_t = NativeGradCAM(model)
        mt, _ = gc_t.generate(vt, geo_d, label, use_unet=True)
        gc_t.remove_hooks()
        maps.append(mt)
        del gc_t, vt, mt
        torch.cuda.empty_cache()

    model.eval()
    stab_map   = np.stack(maps).std(axis=0)
    mean_stab  = float(stab_map.mean())
    del maps
    torch.cuda.empty_cache()
    return stab_map, mean_stab


# ── Localisation ───────────────────────────────────────────────────────────

def compute_localisation(combined, geo_224):
    """IoU and Dice between saliency map and vessel map."""
    sal_bin = (combined > np.percentile(combined, 80)).astype(np.float32)
    geo_bin = (geo_224 > 0.3).astype(np.float32)
    inter   = np.logical_and(sal_bin, geo_bin).sum()
    union   = np.logical_or(sal_bin, geo_bin).sum()
    iou     = inter / (union + 1e-8)
    dice    = (2 * inter) / (sal_bin.sum() + geo_bin.sum() + 1e-8)
    return iou, dice


# ── Z-scores ───────────────────────────────────────────────────────────────

def compute_zscores(model, vis, geo, label):
    """Get raw metrics and Z-scores for a single sample."""
    vis_d, geo_d = vis.to(device), geo.to(device)
    with torch.no_grad(), _autocast_ctx():
        logits, metrics, _ = model(vis_d, geo_d, use_unet=True)
        conf = F.softmax(logits.float(), dim=1)[0, label].item()
    met = metrics.squeeze().float().cpu().numpy()
    z   = (met - mean_normal) / std_normal
    return met, z, conf


# ── Public pipeline entry-point ────────────────────────────────────────────

def run_gradcam_pipeline(
    model: torch.nn.Module,
    img_vis: torch.Tensor,
    img_geo: torch.Tensor,
    target_class: int,
    use_unet: bool = True,
) -> str:
    """
    Compute attention-guided saliency overlay and return as base64-encoded PNG.

    Used by both FastAPI (main.py) and Gradio (app.py).
    """
    gcam, cafm_norm, combined, geo_224, img_np = compute_saliency(
        model, img_vis, img_geo, target_class
    )

    cam_uint8 = np.uint8(255 * combined)
    _, mask   = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    heatmap            = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap[mask == 0] = 0

    orig_uint8 = np.uint8(255 * img_np)
    orig_bgr   = cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR)

    overlay     = cv2.addWeighted(heatmap, 0.55, orig_bgr, 0.45, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    buf = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
