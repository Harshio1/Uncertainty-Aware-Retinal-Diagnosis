class NativeGradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self.handles     = []
        target_layer = self.model.vis_backbone.blocks[-1]
        self.handles.append(
            target_layer.register_forward_hook(self._save_activation))
        self.handles.append(
            target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, m, i, o): self.activations = o
    def _save_gradient(self, m, gi, go): self.gradients   = go[0]

    def generate(self, img_vis, img_geo, target_class, use_unet=True):
        self.model.zero_grad()
        logits, _, attn = self.model(img_vis, img_geo, use_unet=use_unet)
        logits[0, target_class].backward(retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam     = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode='bilinear',
                                 align_corners=False)
        cam     = cam.squeeze().cpu().detach().numpy()
        cmin, cmax = cam.min(), cam.max()
        cam = (cam - cmin) / (cmax - cmin) if cmax - cmin > 1e-8 else np.zeros_like(cam)
        return cam, attn

    def remove_hooks(self):
        for h in self.handles: h.remove()
        self.handles = []


# ─────────────────────────────────────────────────────────────────────────────
# 2g. DiagnosticMetricsExtractor v6  (DISPLAY ONLY — no learnable weights)
# ─────────────────────────────────────────────────────────────────────────────
# Tortuosity approach (both metrics now fully Sobel-based on soft map):
#
#   Avg Tortuosity  = MEDIAN  Sobel gradient magnitude in vessel region (>0.2)
#   Max Tortuosity  = 90th-pct Sobel gradient magnitude in vessel region (>0.2)
#
# Why Sobel on the soft probability map is better than Laplacian on skeleton:
#   - Soft map is continuous float32 → Sobel produces real-valued gradients
#     with genuine per-scan variation (reflecting vessel sharpness / curvature).
#   - Binary skeleton → Laplacian yields integers {0,1,2,3,4}; averaging
#     over hundreds of pixels collapses variance to near-zero (~0.04 std).
#   - Median (50th-pct) gives robust central tendency for avg tortuosity;
#     90th-pct captures the most tortuous vessel segments for max tortuosity.
#   - Both are computed from the same gmag tensor → consistent, efficient.
#
# Skeleton is still computed for Endpoints and Branch Length.
# laplacian_kernel buffer is removed (no longer needed here).
# ─────────────────────────────────────────────────────────────────────────────