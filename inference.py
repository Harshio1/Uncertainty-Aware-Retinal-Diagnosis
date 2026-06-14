"""
inference.py — Model engine: MC-Dropout inference with temperature scaling.

KEY FIXES:
  1. Uses DiagnosticMetricsExtractor (standalone) instead of rm from model's
     internal geo_encoder. The model's internal path uses 128x128 soft-sigmoid
     which was causing branching_index = 0.
  2. Binary threshold lowered to 0.3 to capture diffuse UNet soft maps.
  3. Branching index uses scipy.ndimage.label for real connected components:
     BI = N_components / (total_vessel_pixels + eps)  — exact proposal formula.
  4. Device-conditional autocast (CPU-safe for HF Spaces).
"""

import os
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage as ndi

from model import FinalPatentArchitecture

CLASSES = ["NORMAL", "CNV", "DME", "DRUSEN"]


def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


# ── Standalone DiagnosticMetricsExtractor ─────────────────────────────────

class DiagnosticMetricsExtractor(nn.Module):
    """
    Standalone metric extractor for display/reporting.
    Matches notebook Cell 18 logic exactly.

    Metric order:
      0: density          — fraction of foreground pixels
      1: fractal_dim      — box-counting fractal dimension
      2: lacunarity       — variance/mean^2 texture measure
      3: avg_tortuosity   — median Sobel gradient in vessel region
      4: max_tortuosity   — 90th-pct Sobel gradient in vessel region
      5: branching_index  — N_components / vessel_pixels (proposal formula)
      6: endpoint_count   — skeleton pixels with degree=1
      7: branch_length    — total skeleton pixel count
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            'neighbor_kernel',
            torch.tensor([[[[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]]]])
        )
        self.register_buffer(
            'sobel_x',
            torch.tensor([[[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]]])
        )
        self.register_buffer(
            'sobel_y',
            torch.tensor([[[[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]])
        )

    def _skeletonize(self, mask, iters=25):
        thin = mask.clone()
        for _ in range(iters):
            nb  = F.conv2d(thin, self.neighbor_kernel, padding=1) * thin
            ero = 1.0 - F.max_pool2d(1.0 - thin, kernel_size=3, stride=1, padding=1)
            bnd = ((thin - ero) > 0.5).float()
            rem = bnd * (nb >= 2).float() * (nb <= 6).float()
            thin = (thin * (1.0 - rem) > 0.5).float()
            if rem.sum() == 0:
                break
        return thin

    @torch.no_grad()
    def forward(self, soft_map: torch.Tensor) -> np.ndarray:
        """
        Args:
            soft_map: [B, 1, H, W] float32 — UNet vessel probability map [0,1]
        Returns:
            metrics: [B, 8] numpy array
        """
        eps = 1e-6
        B   = soft_map.shape[0]

        # Upsample to 256x256
        sm = F.interpolate(soft_map.float(), size=(256, 256),
                           mode='bilinear', align_corners=False)

        # Threshold 0.3 — captures diffuse UNet soft maps where most vessel
        # pixels sit between 0.3-0.5 (at 0.5 the binary map is nearly empty)
        binary = (sm > 0.3).float()

        # Density
        density = binary.view(B, -1).mean(dim=1)

        # Fractal Dimension + Lacunarity
        scales = [1, 2, 4, 8, 16]
        bcs, lacs = [], []
        for s in scales:
            if s == 1:
                Ns = binary.view(B, -1).sum(1)
                ap = binary.view(B, -1)
            else:
                Ns = F.max_pool2d(binary, s, s).view(B, -1).sum(1)
                ap = F.avg_pool2d(binary, s, s).view(B, -1)
            bcs.append(Ns)
            mm = ap.mean(1)
            vm = ap.var(1, unbiased=False)
            lacs.append(vm / (mm ** 2 + eps))

        bct = torch.stack(bcs, dim=1)
        lis = torch.tensor(
            [-math.log(s + eps) for s in scales], device=sm.device
        ).unsqueeze(0).expand(B, -1)
        lN  = torch.log(bct + eps)
        xm, ym = lis.mean(1, keepdim=True), lN.mean(1, keepdim=True)
        fd  = ((lis - xm) * (lN - ym)).sum(1) / (((lis - xm) ** 2).sum(1) + eps)
        fd  = torch.clamp(fd, 0.5, 2.0)
        lac = torch.stack(lacs, dim=1).mean(1)

        # Skeleton -> Endpoints + Branch Length
        skeleton   = self._skeletonize(binary, iters=25)
        skel_neigh = F.conv2d(skeleton, self.neighbor_kernel, padding=1) * skeleton
        endpoints  = (skel_neigh == 1.0).float().view(B, -1).sum(1)
        branch_len = skeleton.view(B, -1).sum(1)

        # Branching Index — exact proposal formula:
        # BI = N_connected_components / (sum(B) + eps)
        # Uses scipy.ndimage.label for accurate component counting.
        binary_np = binary.cpu().numpy()  # [B, 1, 256, 256]
        branching_list = []
        for b in range(B):
            bmap          = binary_np[b, 0]
            n_components  = ndi.label(bmap)[1]
            vessel_pixels = float(bmap.sum())
            branching_list.append(float(n_components) / (vessel_pixels + eps))
        branching = torch.tensor(branching_list, dtype=torch.float32,
                                 device=sm.device)

        # Tortuosity (Sobel on soft map)
        gx   = F.conv2d(sm, self.sobel_x, padding=1)
        gy   = F.conv2d(sm, self.sobel_y, padding=1)
        gmag = (gx ** 2 + gy ** 2).sqrt()

        avg_tort_list, max_tort_list = [], []
        for b in range(B):
            vals = gmag[b, 0][sm[b, 0] > 0.2]
            if vals.numel() > 10:
                avg_tort_list.append(torch.quantile(vals, 0.50))
                max_tort_list.append(torch.quantile(vals, 0.90))
            else:
                avg_tort_list.append(torch.tensor(0.0, device=sm.device))
                max_tort_list.append(torch.tensor(0.0, device=sm.device))
        avg_tort = torch.stack(avg_tort_list)
        max_tort = torch.stack(max_tort_list)

        metrics = torch.stack(
            [density, fd, lac, avg_tort, max_tort, branching, endpoints, branch_len],
            dim=1
        )
        return torch.nan_to_num(
            metrics, nan=0., posinf=0., neginf=0.
        ).cpu().numpy()


# ── Model Engine ───────────────────────────────────────────────────────────

class ModelEngine:
    """
    Handles model initialisation and batched inference with uncertainty.
    Uses EXACT notebook configuration: T=20, temperature scaling.
    """

    def __init__(self, weights_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[*] Building FinalPatentArchitecture...")
        self.model = FinalPatentArchitecture(num_classes=4).to(self.device)

        print(f"[*] Loading weights from: {weights_path} (strict=False)")
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        print("[*] Weights loaded successfully (strict=False).")

        self.model.eval()
        self.class_names = CLASSES
        self.temperature = float(os.environ.get("TEMP_SCALE", "1.4060"))
        self.T = 20

        self.diag_extractor = DiagnosticMetricsExtractor().to(self.device)
        self.diag_extractor.eval()
        print("[*] DiagnosticMetricsExtractor ready.")

    def predict_single(
        self, img_vis: torch.Tensor, img_geo: torch.Tensor
    ) -> dict:
        img_vis = img_vis.to(self.device)
        img_geo = img_geo.to(self.device)

        # 1. Vessel map — deterministic, UNet has no dropout
        with torch.no_grad():
            self.model.unet.eval()
            vessel_map = self.model.unet(img_geo)
            print(f"[*] Vessel map — mean: {vessel_map.mean().item():.4f}, "
                  f"max: {vessel_map.max().item():.4f}")

        # 2. MC-Dropout: model.train() activates classifier Dropout only
        self.model.train()
        self.model.geo_encoder.eval()
        self.model.unet.eval()

        all_probs = []
        with torch.no_grad(), _autocast_ctx(self.device):
            for _ in range(self.T):
                logits, _, _ = self.model(img_vis, vessel_map, use_unet=False)
                probs = F.softmax(logits / self.temperature, dim=1)
                all_probs.append(probs.unsqueeze(0))

        # 3. Aggregate MC passes
        all_probs_t      = torch.cat(all_probs, dim=0)
        mean_probs       = all_probs_t.mean(dim=0).squeeze(0).cpu().numpy()
        var_probs        = all_probs_t.var(dim=0).squeeze(0).cpu().numpy()
        pred_idx         = int(np.argmax(mean_probs))
        pred_class_name  = self.class_names[pred_idx]
        pred_confidence  = float(mean_probs[pred_idx])
        pred_uncertainty = float(var_probs[pred_idx])

        # 4. Display metrics via DiagnosticMetricsExtractor (256x256, thresh 0.3)
        with torch.no_grad():
            diag_metrics = self.diag_extractor(vessel_map.float())  # [1, 8]

        metric_names = [
            "density", "fractal_dim", "lacunarity",
            "avg_tortuosity", "max_tortuosity",
            "branching_index", "endpoint_count", "branch_length",
        ]
        metrics_dict = {
            mname: float(diag_metrics[0][i])
            for i, mname in enumerate(metric_names)
        }
        print("[*] Metrics:", {k: round(v, 4) for k, v in metrics_dict.items()})

        return {
            "prediction":        pred_class_name,
            "confidence":        pred_confidence,
            "uncertainty":       pred_uncertainty,
            "all_probabilities": {self.class_names[i]: float(mean_probs[i]) for i in range(4)},
            "vascular_metrics":  metrics_dict,
        }
