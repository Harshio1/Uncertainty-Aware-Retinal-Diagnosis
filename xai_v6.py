def compute_saliency(model, vis, geo, label):
    """
    Returns: gcam, cafm, combined (additive fusion), vessel_map, img_display

    Combined saliency: S = alpha * L2norm(Grad-CAM) + (1-alpha) * L2norm(CAFM)
    alpha=0.6 gives visual-stream priority (higher spatial resolution).
    """
    vis_g = vis.clone().to(device).requires_grad_(True)
    geo_d = geo.to(device)
    gc_obj = NativeGradCAM(model)
    gcam, attn = gc_obj.generate(vis_g, geo_d, label, use_unet=True)
    gc_obj.remove_hooks()

    # CAFM attention: [B, num_heads, 1, 49] → mean over heads → [49] → [7,7]
    cafm_raw = attn[0].mean(dim=0).view(7, 7).cpu().detach().numpy()
    cafm = cv2.resize(cafm_raw, (224, 224), interpolation=cv2.INTER_CUBIC)
    cafm = np.clip(cafm, 0, None)  # ReLU — remove negative interpolation artifacts

    # L2-normalize both maps to [0, 1] range
    def l2_normalize(m):
        m_min, m_max = m.min(), m.max()
        if m_max - m_min < 1e-8:
            return np.zeros_like(m)
        return (m - m_min) / (m_max - m_min)

    gcam_norm = l2_normalize(gcam)
    cafm_norm = l2_normalize(cafm)

    # Weighted additive fusion (alpha=0.6 for visual-dominant)
    # Patent language: "The attention-guided saliency map S is computed as a
    # weighted combination of visual gradient saliency and cross-modal attention,
    # wherein the visual component receives higher weight due to superior
    # spatial resolution."
    alpha = 0.6
    combined = alpha * gcam_norm + (1 - alpha) * cafm_norm
    combined = l2_normalize(combined)  # Re-normalize to [0,1]

    # U-Net vessel probability map
    with torch.no_grad(), torch.amp.autocast('cuda'):
        vmap = model.unet(geo_d)
    geo_224 = F.interpolate(vmap, (224, 224), mode='bilinear',
                             align_corners=False).squeeze().cpu().float().numpy()

    # Denormalize input for display
    img_np = vis.squeeze().cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np * IMG_STD + IMG_MEAN, 0, 1)

    return gcam, cafm_norm, combined, geo_224, img_np



def compute_faithfulness(model, vis, geo, saliency, label, steps=10):
    """
    Dual-stream faithfulness: perturbs BOTH visual and geometric inputs.

    Deletion: Progressively blur most-salient regions in BOTH streams.
              Ideal: confidence drops monotonically → low AUC.
    Insertion: Progressively reveal most-salient regions in BOTH streams.
              Ideal: confidence rises monotonically → high AUC.
    """
    vis_d, geo_d = vis.to(device), geo.to(device)

    # --- Visual stream: denormalize for manipulation ---
    img_np = np.clip(vis.squeeze().cpu().numpy().transpose(1, 2, 0)
                     * IMG_STD + IMG_MEAN, 0, 1).astype(np.float32)
    blurred_vis = cv2.GaussianBlur(img_np, (51, 51), 0)

    # --- Geometric stream: get raw CLAHE input for manipulation ---
    geo_np = geo.squeeze().cpu().numpy()  # [1, 512, 512] → [512, 512]
    blurred_geo = cv2.GaussianBlur(geo_np, (51, 51), 0)

    # --- Saliency ordering (on 224×224 visual resolution) ---
    sort_idx_vis = np.argsort(saliency.flatten())[::-1]
    step_size_vis = max(1, len(sort_idx_vis) // steps)

    # --- Upscale saliency to geo resolution (512×512) for geo perturbation ---
    saliency_geo = cv2.resize(saliency, (geo_np.shape[1], geo_np.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
    sort_idx_geo = np.argsort(saliency_geo.flatten())[::-1]
    step_size_geo = max(1, len(sort_idx_geo) // steps)

    def vis_to_tensor(arr):
        t = (arr - IMG_MEAN) / IMG_STD
        return torch.tensor(t.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    def geo_to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): m.eval()

    del_p, ins_p = [], []

    for s in range(steps + 1):
        # Visual pixels to modify
        px_vis = sort_idx_vis[:s * step_size_vis]
        # Geometric pixels to modify (at 512×512 resolution)
        px_geo = sort_idx_geo[:s * step_size_geo]

        # --- DELETION: original → progressively blur salient regions ---
        d_vis = img_np.copy().reshape(-1, 3)
        d_vis[px_vis] = blurred_vis.reshape(-1, 3)[px_vis]

        d_geo = geo_np.copy().flatten()
        d_geo[px_geo] = blurred_geo.flatten()[px_geo]
        d_geo = d_geo.reshape(geo_np.shape)

        # --- INSERTION: blurred → progressively reveal salient regions ---
        i_vis = blurred_vis.copy().reshape(-1, 3)
        i_vis[px_vis] = img_np.reshape(-1, 3)[px_vis]

        i_geo = blurred_geo.copy().flatten()
        i_geo[px_geo] = geo_np.flatten()[px_geo]
        i_geo = i_geo.reshape(geo_np.shape)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            d_out, _, _ = model(vis_to_tensor(d_vis.reshape(224, 224, 3)),
                                geo_to_tensor(d_geo), use_unet=True)
            i_out, _, _ = model(vis_to_tensor(i_vis.reshape(224, 224, 3)),
                                geo_to_tensor(i_geo), use_unet=True)

        del_p.append(F.softmax(d_out.float(), 1)[0, label].item())
        ins_p.append(F.softmax(i_out.float(), 1)[0, label].item())

    fracs = np.linspace(0, 1, steps + 1)
    return auc(fracs, del_p), auc(fracs, ins_p), fracs, del_p, ins_p


def compute_stability(model, vis, geo, label, T=10):
    """MC-Dropout Grad-CAM stability with aggressive memory cleanup."""
    geo_d = geo.to(device)
    # Enable dropout
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    maps = []
    for _ in range(T):
        # Create fresh GradCAM instance for each pass
        vt = vis.clone().to(device).requires_grad_(True)
        gc_t = NativeGradCAM(model)
        mt, _ = gc_t.generate(vt, geo_d, label, use_unet=True)
        gc_t.remove_hooks()
        maps.append(mt)

        # Delete everything related to this pass
        del gc_t, vt, mt
        torch.cuda.empty_cache()

    # Restore model to eval mode
    model.eval()

    # Compute stability
    stab_map = np.stack(maps).std(axis=0)
    mean_stab = float(stab_map.mean())

    # Clean up maps
    del maps
    torch.cuda.empty_cache()
    return stab_map, mean_stab


def compute_localisation(combined, geo_224):
    """IoU and Dice between saliency and vessel map"""
    sal_bin = (combined > np.percentile(combined, 80)).astype(np.float32)
    geo_bin = (geo_224 > 0.3).astype(np.float32)
    inter = np.logical_and(sal_bin, geo_bin).sum()
    union = np.logical_or(sal_bin, geo_bin).sum()
    iou  = inter / (union + 1e-8)
    dice = (2 * inter) / (sal_bin.sum() + geo_bin.sum() + 1e-8)
    return iou, dice


def compute_zscores(model, vis, geo, label):
    """Get metrics and Z-scores for a single sample"""
    vis_d, geo_d = vis.to(device), geo.to(device)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        logits, metrics, _ = model(vis_d, geo_d, use_unet=True)
        conf = F.softmax(logits.float(), dim=1)[0, label].item()
    met = metrics.squeeze().float().cpu().numpy()
    z = (met - mean_normal) / std_normal
    return met, z, conf


# ─────────────────────────────────────────────────────────────────────────────
# 7. RUN XAI FOR ALL CLASSES (N samples each)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[*] Running full XAI pipeline (4 classes x {N_SAMPLES_PER_CLASS} samples)...")

xai_results = {}  # {cls_idx: {aggregated results}}

for cls_idx in range(4):
    cls_name = classes[cls_idx]
    samples = class_samples[cls_idx]
    print(f"\n  === {cls_name} ({len(samples)} samples) ===")

    # Store per-sample results
    all_del_auc, all_ins_auc = [], []
    all_iou, all_dice = [], []
    all_stab = []
    all_z = []
    all_met = []
    all_conf = []

    # For Figure 1: keep the BEST (highest confidence) sample's visuals
    best_idx = 0  # first sample (already sorted by collection order)
    best_vis_data = None

    for si, (vis, geo, label, conf) in enumerate(samples):
        print(f"    Sample {si+1}/{len(samples)} (conf={conf:.3f})...", end=" ")

        # Saliency (only store visuals for best sample)
        gcam, cafm, combined, geo_224, img_np = compute_saliency(
            final_model, vis, geo, label)

        if si == best_idx:
            best_vis_data = {
                'img': img_np, 'gcam': gcam, 'cafm': cafm,
                'combined': combined, 'geo_224': geo_224
            }

        # Localisation
        iou, dice = compute_localisation(combined, geo_224)
        all_iou.append(iou); all_dice.append(dice)

        # Faithfulness
        d_auc, i_auc, fracs, del_p, ins_p = compute_faithfulness(
            final_model, vis, geo, combined, label, steps=FAITH_STEPS)
        all_del_auc.append(d_auc); all_ins_auc.append(i_auc)

        # Keep curves from best sample for plotting
        if si == best_idx:
            best_vis_data['fracs'] = fracs
            best_vis_data['del_p'] = del_p
            best_vis_data['ins_p'] = ins_p

        # Stability
        stab_map, mean_stab = compute_stability(
            final_model, vis, geo, label, T=T_STABILITY)
        all_stab.append(mean_stab)
        if si == best_idx:
            best_vis_data['stab_map'] = stab_map

        # Z-scores
        met, z, c = compute_zscores(final_model, vis, geo, label)
        all_z.append(z); all_met.append(met); all_conf.append(c)

        print(f"IoU={iou:.3f} Del={d_auc:.3f} Ins={i_auc:.3f} Stab={mean_stab:.4f}")

    # Aggregate
    xai_results[cls_idx] = {
        'name': cls_name,
        'n_samples': len(samples),
        'vis': best_vis_data,
        # Averaged metrics with std
        'del_auc_mean': np.mean(all_del_auc), 'del_auc_std': np.std(all_del_auc),
        'ins_auc_mean': np.mean(all_ins_auc), 'ins_auc_std': np.std(all_ins_auc),
        'iou_mean': np.mean(all_iou), 'iou_std': np.std(all_iou),
        'dice_mean': np.mean(all_dice), 'dice_std': np.std(all_dice),
        'stab_mean': np.mean(all_stab), 'stab_std': np.std(all_stab),
        'conf_mean': np.mean(all_conf), 'conf_std': np.std(all_conf),
        'z_mean': np.stack(all_z).mean(axis=0),
        'z_std': np.stack(all_z).std(axis=0),
        'met_mean': np.stack(all_met).mean(axis=0),
    }

gc.collect(); torch.cuda.empty_cache()
print("\n[*] All XAI computations complete.")


# =============================================================================
# FIGURE 1: VISUAL SALIENCY MAPS (4 rows x 5 cols)
# =============================================================================
print("\n[*] Rendering Figure 1: Visual Saliency Maps...")

cls_colors = {'NORMAL': '#2CA02C', 'CNV': '#D62728', 'DME': '#1F77B4', 'DRUSEN': '#FF7F0E'}

fig1, axes1 = plt.subplots(4, 5, figsize=(24, 19), facecolor='white')
fig1.suptitle('Figure 1: Explainability Analysis — Visual Saliency Maps\n'
              'Representative scan per class (best confidence from N=5 evaluated)',
              fontsize=15, fontweight='bold', y=0.99)

col_titles = ['CLAHE Input', 'U-Net Vessel Map', 'Grad-CAM',
              'CAFM Attention', 'Attention-Guided Saliency']

for row, cls_idx in enumerate(range(4)):
    d = xai_results[cls_idx]
    v = d['vis']

    # Col 0: Input
    axes1[row, 0].imshow(v['img'])
    axes1[row, 0].set_ylabel(
        f"{d['name']}\n{d['conf_mean']:.1%} conf\n(N={d['n_samples']})",
        fontsize=12, fontweight='bold', rotation=0, labelpad=85, va='center',
        color=cls_colors[d['name']])

    # Col 1: Vessel map
    im1 = axes1[row, 1].imshow(v['geo_224'], cmap='plasma', vmin=0, vmax=1)
    if row == 0:
        plt.colorbar(im1, ax=axes1[row, 1], fraction=0.046, pad=0.04, label='p(vessel)')

    # Col 2: Grad-CAM
    axes1[row, 2].imshow(v['img'])
    axes1[row, 2].imshow(v['gcam'], cmap='jet', alpha=0.55)

    # Col 3: CAFM
    axes1[row, 3].imshow(v['img'])
    axes1[row, 3].imshow(v['cafm'], cmap='magma', alpha=0.60)

    # Col 4: Combined
    axes1[row, 4].imshow(v['img'])
    axes1[row, 4].imshow(v['combined'], cmap='inferno', alpha=0.55)
    for col in range(5):
        axes1[row, col].axis('off')
        if row == 0:
            axes1[row, col].set_title(col_titles[col], fontsize=12,
                                       fontweight='bold', pad=10)

plt.tight_layout(rect=[0.08, 0.02, 1.0, 0.95])
plt.savefig('xai_fig1_saliency.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[*] Saved: xai_fig1_saliency.png")
plt.show()


# =============================================================================
# FIGURE 2: FAITHFULNESS CURVES (2x2, one per class)
# =============================================================================
print("[*] Rendering Figure 2: Faithfulness Evaluation...")

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11), facecolor='white')
fig2.suptitle('Figure 2: Faithfulness Evaluation — Deletion & Insertion AUC\n'
              f'Mean AUC \u00b1 std computed over N={N_SAMPLES_PER_CLASS} samples per class',
              fontsize=14, fontweight='bold', y=1.0)

for idx, cls_idx in enumerate(range(4)):
    d = xai_results[cls_idx]
    v = d['vis']
    ax = axes2[idx // 2, idx % 2]

    ax.plot(v['fracs'] * 100, v['del_p'], 'o-', color='#D62728', lw=2, ms=5,
            label=f"Deletion AUC = {d['del_auc_mean']:.3f}\u00b1{d['del_auc_std']:.3f}")
    ax.plot(v['fracs'] * 100, v['ins_p'], 's-', color='#2CA02C', lw=2, ms=5,
            label=f"Insertion AUC = {d['ins_auc_mean']:.3f}\u00b1{d['ins_auc_std']:.3f}")
    ax.fill_between(v['fracs'] * 100, v['del_p'], alpha=0.08, color='#D62728')
    ax.fill_between(v['fracs'] * 100, v['ins_p'], alpha=0.08, color='#2CA02C')
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.05)
    ax.set_title(f'{d["name"]} (conf: {d["conf_mean"]:.1%}\u00b1{d["conf_std"]:.1%})',
                  fontsize=12, fontweight='bold', color=cls_colors[d['name']])
    ax.set_xlabel('Salient pixels modified (%)', fontsize=10)
    ax.set_ylabel(f'p(y = {d["name"]} | x)', fontsize=10)
    ax.legend(fontsize=9, loc='center right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('xai_fig2_faithfulness.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[*] Saved: xai_fig2_faithfulness.png")
plt.show()


# =============================================================================
# FIGURE 3: Z-SCORE PROFILES (2x2, one per class)
# =============================================================================
print("[*] Rendering Figure 3: Vascular Z-Score Profiles...")

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 11), facecolor='white')
fig3.suptitle('Figure 3: Vascular Metric Z-Score Profiles (vs. NORMAL baseline)\n'
              f'z = (metric - mean_NORMAL) / std_NORMAL   |   '
              f'Mean \u00b1 std over N={N_SAMPLES_PER_CLASS} samples   |   '
              f'Red: |z| > 2 (clinically significant)',
              fontsize=13, fontweight='bold', y=1.01)

for idx, cls_idx in enumerate(range(4)):
    d = xai_results[cls_idx]
    ax = axes3[idx // 2, idx % 2]
    z_vals = d['z_mean']
    z_errs = d['z_std']

    colors_z = ['#D62728' if abs(z) > 2 else '#999999' for z in z_vals]
    bars = ax.barh(metric_short, z_vals, xerr=z_errs, color=colors_z,
                    height=0.6, edgecolor='white', linewidth=0.5,
                    capsize=3, error_kw={'lw': 1.2, 'capthick': 1.2})
    ax.axvline(2, color='#D62728', ls='--', lw=1, alpha=0.6)
    ax.axvline(-2, color='#D62728', ls='--', lw=1, alpha=0.6)
    ax.axvline(0, color='#888888', ls='-', lw=0.6, alpha=0.5)

    for bar, z in zip(bars, z_vals):
        offset = 0.2 if z >= 0 else -0.2
        ax.text(z + offset, bar.get_y() + bar.get_height() / 2,
                f'{z:+.2f}', va='center',
                ha='left' if z >= 0 else 'right',
                fontsize=8.5, fontweight='bold', color='#222222')

    ax.set_title(f'{d["name"]}', fontsize=12, fontweight='bold',
                  color=cls_colors[d['name']])
    ax.set_xlabel('Z-Score (std deviations from NORMAL)', fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('xai_fig3_zscores.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[*] Saved: xai_fig3_zscores.png")
plt.show()


# =============================================================================
# FIGURE 4: EXPLANATION STABILITY (1x4)
# =============================================================================
print("[*] Rendering Figure 4: Explanation Stability...")

fig4, axes4 = plt.subplots(1, 4, figsize=(22, 5.5), facecolor='white')
fig4.suptitle(f'Figure 4: Explanation Stability — MC-Dropout Grad-CAM (T={T_STABILITY})\n'
              f'Per-pixel std of stochastic saliency maps   |   '
              f'Mean sigma \u00b1 std over N={N_SAMPLES_PER_CLASS} samples',
              fontsize=13, fontweight='bold', y=1.03)

for idx, cls_idx in enumerate(range(4)):
    d = xai_results[cls_idx]
    v = d['vis']
    ax = axes4[idx]
    vmax = max(0.05, v['stab_map'].max())
    im = ax.imshow(v['stab_map'], cmap='hot', vmin=0, vmax=vmax)
    ax.set_title(f"{d['name']}\n"
                 f"sigma = {d['stab_mean']:.4f}\u00b1{d['stab_std']:.4f}",
                 fontsize=11, fontweight='bold', color=cls_colors[d['name']])
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pixel std')

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig('xai_fig4_stability.png', dpi=300, bbox_inches='tight', facecolor='white')
print("[*] Saved: xai_fig4_stability.png")
plt.show()


# =============================================================================
# JSON REPORT
# =============================================================================
print("\n[*] Writing xai_report.json...")

report = {
    "model": "CAFM-OCT-v1.0",
    "xai_config": {
        "n_samples_per_class": N_SAMPLES_PER_CLASS,
        "stability_T": T_STABILITY,
        "faithfulness_steps": FAITH_STEPS,
        "saliency_method": "Attention-Guided Grad-CAM (Grad-CAM x CAFM)",
        "faithfulness_method": "Deletion/Insertion with Gaussian blur baseline",
        "stability_method": "MC-Dropout stochastic Grad-CAM"
    },
    "baseline": {
        "reference_class": "NORMAL",
        "justification": ("Clinical standard: pathological deviations measured "
                          "against healthy controls. Matches FDA-cleared OCT "
                          "devices (Zeiss Cirrus, Heidelberg Spectralis)."),
        "n_normal_scans": int(normal_mask.sum()),
    },
    "per_class": {}
}

for cls_idx in range(4):
    d = xai_results[cls_idx]
    report["per_class"][d['name']] = {
        "n_samples": d['n_samples'],
        "confidence": f"{d['conf_mean']:.4f} +/- {d['conf_std']:.4f}",
        "faithfulness": {
            "deletion_auc": f"{d['del_auc_mean']:.4f} +/- {d['del_auc_std']:.4f}",
            "insertion_auc": f"{d['ins_auc_mean']:.4f} +/- {d['ins_auc_std']:.4f}",
        },
        "localisation": {
            "iou": f"{d['iou_mean']:.4f} +/- {d['iou_std']:.4f}",
            "dice": f"{d['dice_mean']:.4f} +/- {d['dice_std']:.4f}",
        },
        "stability": {
            "mean_pixel_sigma": f"{d['stab_mean']:.4f} +/- {d['stab_std']:.4f}",
        },
        "vascular_z_scores": {
            metric_names[i]: {
                "z_mean": round(float(d['z_mean'][i]), 4),
                "z_std": round(float(d['z_std'][i]), 4),
                "flag": "HIGH" if d['z_mean'][i] > 2 else "LOW" if d['z_mean'][i] < -2 else "NORMAL"
            } for i in range(8)
        }
    }

with open('xai_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print("[*] Saved: xai_report.json")


# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 85)
print(f"  XAI SUMMARY — ALL CLASSES (N={N_SAMPLES_PER_CLASS} per class, mean +/- std)")
print("=" * 85)
print(f"  {'Class':<10} {'Confidence':>14} {'IoU':>14} {'Del AUC':>14} {'Ins AUC':>14} {'Stability':>14}")
print("  " + "-" * 80)
for cls_idx in range(4):
    d = xai_results[cls_idx]
    print(f"  {d['name']:<10} "
          f"{d['conf_mean']:.3f}\u00b1{d['conf_std']:.3f}  "
          f"{d['iou_mean']:.3f}\u00b1{d['iou_std']:.3f}  "
          f"{d['del_auc_mean']:.3f}\u00b1{d['del_auc_std']:.3f}  "
          f"{d['ins_auc_mean']:.3f}\u00b1{d['ins_auc_std']:.3f}  "
          f"{d['stab_mean']:.4f}\u00b1{d['stab_std']:.4f}")
print("=" * 85)

print("\n[*] ALL XAI COMPLETE.")
print("[*] Output figures (SEPARATE, publication-ready):")
print("      xai_fig1_saliency.png     — 4x5 visual saliency grid")
print("      xai_fig2_faithfulness.png  — deletion/insertion curves per class")
print("      xai_fig3_zscores.png       — vascular Z-score profiles per class")
print("      xai_fig4_stability.png     — MC-Dropout stability maps per class")
print("      xai_report.json            — structured JSON (all classes, all metrics)")
--- CELL 20 ---
# ==============================================================================
# CELL 16 — PUBLICATION-GRADE XAI SUITE (Multi-Sample, Separate Figures)
# ==============================================================================
#
# KEY IMPROVEMENTS OVER v3: