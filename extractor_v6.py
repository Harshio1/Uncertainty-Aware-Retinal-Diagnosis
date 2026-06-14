class DiagnosticMetricsExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('neighbor_kernel',
            torch.tensor([[[[1.,1.,1.],[1.,0.,1.],[1.,1.,1.]]]]));
        # Sobel kernels — used for BOTH Avg and Max Tortuosity
        self.register_buffer('sobel_x',
            torch.tensor([[[[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]]]));
        self.register_buffer('sobel_y',
            torch.tensor([[[[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]]]]));

    def _skeletonize(self, mask, iters=25):
        thin = mask.clone()
        for _ in range(iters):
            nb  = F.conv2d(thin, self.neighbor_kernel, padding=1) * thin
            ero = 1.0 - F.max_pool2d(1.0-thin, kernel_size=3, stride=1, padding=1)
            bnd = ((thin - ero) > 0.5).float()
            rem = bnd * (nb >= 2).float() * (nb <= 6).float()
            thin = (thin * (1.0 - rem) > 0.5).float()
            if rem.sum() == 0:
                break
        return thin

    @torch.no_grad()
    def forward(self, soft_map):
        """
        Args:
            soft_map : [B, 1, H, W] float32 — U-Net vessel probability map [0,1].
                       Cast with .float() before calling (autocast safety).
        Returns:
            metrics  : [B, 8] numpy array — same order as metric_names.
        """
        eps = 1e-6
        B   = soft_map.shape[0]

        # Upsample to 256×256 for better vessel resolution
        sm     = F.interpolate(soft_map, size=(256, 256),
                               mode='bilinear', align_corners=False)
        binary = (sm > 0.5).float()

        # ── Density ──────────────────────────────────────────────────────────
        density = binary.view(B, -1).mean(dim=1)

        # ── Fractal Dimension (box-counting, scales 1,2,4,8,16) ──────────────
        scales, bcs, lacs = [1, 2, 4, 8, 16], [], []
        for s in scales:
            if s == 1:
                Ns = binary.view(B, -1).sum(1)
                ap = binary.view(B, -1)
            else:
                Ns = F.max_pool2d(binary, s, s).view(B, -1).sum(1)
                ap = F.avg_pool2d(binary, s, s).view(B, -1)
            bcs.append(Ns)
            mm = ap.mean(1); vm = ap.var(1, unbiased=False)
            lacs.append(vm / (mm**2 + eps))
        bct = torch.stack(bcs, dim=1)
        lis = torch.tensor([-math.log(s+eps) for s in scales],
                            device=sm.device).unsqueeze(0).expand(B, -1)
        lN  = torch.log(bct + eps)
        xm, ym = lis.mean(1, keepdim=True), lN.mean(1, keepdim=True)
        fd  = ((lis-xm)*(lN-ym)).sum(1) / (((lis-xm)**2).sum(1) + eps)
        fd  = torch.clamp(fd, 0.5, 2.0)
        lac = torch.stack(lacs, dim=1).mean(1)

        # ── Skeleton — for Endpoints and Branch Length only ───────────────────
        skeleton   = self._skeletonize(binary, iters=25)
        skel_neigh = F.conv2d(skeleton, self.neighbor_kernel, padding=1) * skeleton
        endpoints  = (skel_neigh == 1.0).float().view(B, -1).sum(1)
        branch_len = skeleton.view(B, -1).sum(1)

        # ── Sobel gradient magnitude on SOFT map ──────────────────────────────
        # Computed once, used for both Avg and Max Tortuosity.
        # Vessel region mask: soft probability > 0.2 (includes uncertain edges).
        gx   = F.conv2d(sm, self.sobel_x, padding=1)
        gy   = F.conv2d(sm, self.sobel_y, padding=1)
        gmag = (gx**2 + gy**2).sqrt()          # [B, 1, 256, 256] continuous

        avg_tort_list, max_tort_list = [], []
        for b in range(B):
            vals = gmag[b, 0][sm[b, 0] > 0.2]  # pixels in vessel region
            if vals.numel() > 10:
                # Avg Tortuosity = median gradient magnitude (robust central tendency)
                avg_tort_list.append(torch.quantile(vals, 0.50))
                # Max Tortuosity = 90th-pct gradient magnitude (peak curvature)
                max_tort_list.append(torch.quantile(vals, 0.90))
            else:
                avg_tort_list.append(torch.tensor(0.0, device=sm.device))
                max_tort_list.append(torch.tensor(0.0, device=sm.device))
        avg_tort = torch.stack(avg_tort_list)
        max_tort = torch.stack(max_tort_list)

        # ── Branching: vessel pixels with ≥3 vessel neighbours (binary mask) ──
        mask_neigh = F.conv2d(binary, self.neighbor_kernel, padding=1) * binary
        branching  = (mask_neigh >= 3.0).float().view(B, -1).sum(1)

        metrics = torch.stack([density, fd, lac,
                                avg_tort, max_tort,
                                branching, endpoints, branch_len], dim=1)
        return torch.nan_to_num(metrics, nan=0., posinf=0., neginf=0.).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOAD .pth
# ─────────────────────────────────────────────────────────────────────────────
PTH_PATH = '/content/final_patent_architecture.pth'   # ← edit if needed

print("\n[*] Building model and loading weights...")
final_model = FinalPatentArchitecture(num_classes=4).to(device)
state_dict  = torch.load(PTH_PATH, map_location=device)
final_model.load_state_dict(state_dict)
final_model.eval()
print(f"[*] Loaded weights from: {PTH_PATH}")

diag_extractor = DiagnosticMetricsExtractor().to(device)
diag_extractor.eval()

# ─────────────────────────────────────────────────────────────────────────────
# 4. DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[*] Initiating Final Diagnostics Suite...")

# ── A. Feature Extraction ─────────────────────────────────────────────────────
fused_emb_list, diag_labels, diag_metrics, diag_probs = [], [], [], []
activation = {}
hook_handle = final_model.classifier[0].register_forward_hook(
    lambda m, inp, out: activation.__setitem__('fused_vector', inp[0].detach()))

with torch.no_grad():
    for img_vis, img_geo, labels in tqdm(test_loader, desc="Extracting Features"):
        img_vis = img_vis.to(device)
        img_geo = img_geo.to(device)
        with torch.amp.autocast('cuda'):
            # U-Net runs ONCE — reused for model forward and diagnostic extractor
            vessel_map   = final_model.unet(img_geo)                  # [B,1,512,512]
            logits, _, _ = final_model(img_vis, vessel_map, use_unet=False)
            probs        = F.softmax(logits.float(), dim=1)
            # Cast to float32 before extractor (torch.quantile needs float32)
            disp_metrics = diag_extractor(vessel_map.float())          # [B,8] numpy
        fused_emb_list.append(activation['fused_vector'].cpu().numpy())
        diag_labels.extend(labels.cpu().numpy())
        diag_metrics.append(disp_metrics)
        diag_probs.append(probs.cpu().numpy())

hook_handle.remove()

fused_embeddings = np.concatenate(fused_emb_list,  axis=0)
diag_labels      = np.array(diag_labels)
diag_metrics     = np.concatenate(diag_metrics, axis=0)
diag_probs       = np.concatenate(diag_probs,   axis=0)
print(f"[*] Extracted {fused_embeddings.shape[0]} embeddings, dim={fused_embeddings.shape[1]}")

# ── B. t-SNE & Silhouette ─────────────────────────────────────────────────────
print("\n[*] Computing t-SNE Projection...")
tsne          = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(fused_embeddings)
sil_score     = silhouette_score(fused_embeddings, diag_labels)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, ax = plt.subplots(figsize=(9, 7))
palette = sns.color_palette("Set2", n_colors=4)
for i in range(4):
    idx = (diag_labels == i)
    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
               c=[palette[i]], label=classes[i],
               alpha=0.75, edgecolors='w', linewidth=0.5, s=60)
ax.set_title(f"t-SNE of CAFM Fused Embeddings\nSilhouette: {sil_score:.4f}",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("t-SNE Dim 1", fontsize=12)
ax.set_ylabel("t-SNE Dim 2", fontsize=12)
ax.legend(title="Classes", fontsize=11, loc='best', frameon=True, shadow=True)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('tsne_embeddings.png', dpi=300, bbox_inches='tight')
print(f"[*] Silhouette Score: {sil_score:.4f}")
plt.show()

# ── C. Per-Metric Distributions ───────────────────────────────────────────────
print("\n" + "="*60)
print("GEOMETRIC METRIC DISTRIBUTIONS (Mean +/- Std)")
print("="*60)
metric_names = ["Density", "Fractal Dim", "Lacunarity", "Avg Tortuosity",
                "Max Tortuosity", "Branching", "Endpoints", "Branch Length"]
for i, name in enumerate(metric_names):
    print(f"{name:<15}: {diag_metrics[:, i].mean():.4f} +/- {diag_metrics[:, i].std():.4f}")
print()
print("NOTE: Avg Tortuosity = median Sobel gradient in vessel region")
print("      Max Tortuosity = 90th-pct Sobel gradient in vessel region")
print("      Both computed on soft probability map (continuous, float32)")

# ── D. Top-1 & Top-2 Accuracy ─────────────────────────────────────────────────
top1 = top_k_accuracy_score(diag_labels, diag_probs, k=1)
top2 = top_k_accuracy_score(diag_labels, diag_probs, k=2)
print(f"\nTop-1 Accuracy: {top1:.4f}")
print(f"Top-2 Accuracy: {top2:.4f}")

# ── E. Robustness to Segmentation Noise ──────────────────────────────────────
print("\n[*] Testing Robustness under Synthetic Segmentation Noise...")
noise_levels   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
robustness_acc = []

for noise in noise_levels:
    correct, total = 0, 0
    with torch.no_grad():
        for img_vis, img_geo, labels in test_loader:
            img_vis = img_vis.to(device)
            img_geo = img_geo.to(device)
            labels  = labels.to(device)
            noise_mask = (torch.rand_like(img_geo) > noise).float()
            noisy_geo  = img_geo * noise_mask
            with torch.amp.autocast('cuda'):
                logits, _, _ = final_model(img_vis, noisy_geo, use_unet=True)
                preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    acc = correct / total
    robustness_acc.append(acc)
    print(f"  Noise {noise*100:2.0f}% -> Accuracy: {acc:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([n*100 for n in noise_levels], [a*100 for a in robustness_acc],
        'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Noise Level (%)', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Robustness to Geometric Input Noise', fontsize=15)
ax.set_ylim(min(robustness_acc)*100 - 5, 100)
ax.grid(alpha=0.3)
for n, a in zip(noise_levels, robustness_acc):
    ax.annotate(f'{a*100:.1f}%', (n*100, a*100), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('robustness_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# ── F. Efficiency Benchmarks ──────────────────────────────────────────────────
print("\n[*] Running Efficiency Benchmarks...")
dummy_vis = torch.randn(1, 3, 224, 224).to(device)
dummy_geo = torch.randn(1, 1, 512, 512).to(device)

for _ in range(10):
    with torch.no_grad(), torch.amp.autocast('cuda'):
        _ = final_model(dummy_vis, dummy_geo, use_unet=True)

torch.cuda.reset_peak_memory_stats(device)
latencies = []
with torch.no_grad():
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda'):
            _ = final_model(dummy_vis, dummy_geo, use_unet=True)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

mean_lat  = np.mean(latencies)
std_lat   = np.std(latencies)
peak_mem  = torch.cuda.max_memory_allocated(device) / (1024**2)
total_p   = sum(p.numel() for p in final_model.parameters())
trainable = sum(p.numel() for p in final_model.parameters() if p.requires_grad)

print("="*45)
print("EFFICIENCY METRICS")
print("="*45)
print(f"Inference Latency : {mean_lat:.2f} +/- {std_lat:.2f} ms")
print(f"Peak GPU Memory   : {peak_mem:.2f} MB")
print(f"Total Parameters  : {total_p:,}")
print(f"Trainable Params  : {trainable:,}")
print("="*45)

# ── G. Saliency–Vessel Overlap (Grad-CAM ∩ Vessel Map) ────────────────────────
# Computed on first test-batch image (typically NORMAL).
# Use Cell 16 CNV result as primary localization metric.
print("\n[*] Computing Saliency Overlap Score (Grad-CAM ∩ Vessel Map)...")

test_iter = iter(test_loader)
img_vis_s, img_geo_s, target_s = next(test_iter)
img_vis_s    = img_vis_s[0:1].to(device)
img_geo_s    = img_geo_s[0:1].to(device)
target_label = target_s[0].item()
img_vis_s.requires_grad = True

grad_cam_eval        = NativeGradCAM(final_model)
cam_mask_eval, _     = grad_cam_eval.generate(
    img_vis_s, img_geo_s, target_label, use_unet=True)
grad_cam_eval.remove_hooks()

with torch.no_grad(), torch.amp.autocast('cuda'):
    vessel_map_s = final_model.unet(img_geo_s)

vessel_224 = F.interpolate(vessel_map_s, size=(224, 224), mode='bilinear',
                            align_corners=False).squeeze().cpu().float().numpy()

cam_binary    = (cam_mask_eval > np.percentile(cam_mask_eval, 75)).astype(np.float32)
vessel_binary = (vessel_224 > 0.3).astype(np.float32)
intersection  = np.logical_and(cam_binary, vessel_binary).sum()
union         = np.logical_or(cam_binary,  vessel_binary).sum()
overlap_iou   = intersection / (union + 1e-8)
overlap_dice  = (2 * intersection) / (cam_binary.sum() + vessel_binary.sum() + 1e-8)

print(f"[*] Saliency-Vessel IoU:  {overlap_iou:.4f}  (class={classes[target_label]}, sanity check)")
print(f"[*] Saliency-Vessel Dice: {overlap_dice:.4f}  (use Cell 16 CNV result as primary metric)")

print("\n[*] ALL DIAGNOSTICS COMPLETE.")
print("[*] Saved: tsne_embeddings.png, robustness_curve.png")
--- CELL 19 ---
# ==============================================================================
# CELL 16 — PUBLICATION-GRADE XAI SUITE (Multi-Sample, Separate Figures)
# ==============================================================================
#
# KEY IMPROVEMENTS OVER v3:
#   1. N=5 representative scans per class (20 total) — statistically valid
#   2. Metrics (faithfulness, stability, IoU) averaged over N samples with std
#   3. 4 SEPARATE publication figures (not one crowded dashboard):
#        Fig 1: xai_fig1_saliency.png     — 4×5 grid (visual saliency)
#        Fig 2: xai_fig2_faithfulness.png  — deletion/insertion per class
#        Fig 3: xai_fig3_zscores.png       — vascular Z-score profiles
#        Fig 4: xai_fig4_stability.png     — MC-Dropout stability maps
#   4. xai_report.json — structured JSON with per-class averaged metrics
#
# WHY NORMAL BASELINE:
#   In clinical diagnostics, pathological deviation is ALWAYS quantified
#   relative to healthy controls. This is standard practice in:
#   - FDA-cleared OCT devices (Zeiss Cirrus, Heidelberg Spectralis)
#   - OCTA vascular density reports (compared to age-matched normals)
#   - Every published retinal imaging study with Z-score analysis
#   The Z-score z_i = (m_i - mu_NORMAL) / sigma_NORMAL answers:
#   "How many standard deviations from healthy is this scan?"
# ==============================================================================

# ── 0. IMPORTS ─────────────────────────────────────────────────────────────────
import os, gc, glob, math, json, time, warnings, random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from tqdm import tqdm
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')
gc.collect(); torch.cuda.empty_cache()

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'figure.facecolor': 'white',
    'axes.facecolor': '#F8F9FB', 'grid.color': '#DDDDDD',
    'grid.linewidth': 0.5, 'figure.dpi': 100, 'text.usetex': False,
})

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Device: {device}")

IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])

# ── CONFIG ─────────────────────────────────────────────────────────────────────
N_SAMPLES_PER_CLASS = 5      # Number of representative scans per class
T_STABILITY         = 5    # MC-Dropout Grad-CAM passes for stability
FAITH_STEPS         = 10     # Faithfulness deletion/insertion granularity

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────
classes   = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
label_map = {cls: idx for idx, cls in enumerate(classes)}

base_dir = ('/content/dataset/OCT2017/'
            if os.path.exists('/content/dataset/OCT2017/')
            else '/content/dataset/OCT2017 /')

all_paths, all_labels_raw = [], []
for split in ['train', 'val', 'test']:
    for cls in classes:
        paths = glob.glob(os.path.join(base_dir, split, cls, '*.jpeg'))
        all_paths.extend(paths); all_labels_raw.extend([cls] * len(paths))

df = pd.DataFrame({'path': all_paths, 'label': all_labels_raw,
                    'class_idx': [label_map[l] for l in all_labels_raw]})
_, temp_df = train_test_split(df, test_size=0.20, stratify=df['label'], random_state=42)
_, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)
print(f"[*] Test set: {len(test_df)} images")

