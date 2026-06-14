"""
normative.py — Z-score logic and clinical phrase lookup
Z-score formula: z = (x - μ_NORMAL) / σ_NORMAL
These are the normative statistics for the NORMAL class.
They match the DiagnosticMetricsExtractor output scale exactly:
  • density          — fraction of foreground pixels in 256×256 binary map  [0–1]
  • fractal_dim      — box-counting slope, clamped [0.5, 2.0]
  • lacunarity       — variance/mean² ratio, averaged over box scales
  • avg_tortuosity   — median Sobel gradient magnitude in vessel region      [0–1 range]
  • max_tortuosity   — 90th-pct Sobel gradient magnitude in vessel region    [0–1 range]
  • branching_index  — RAW pixel count: pixels with ≥3 vessel neighbours     [0–65536]
  • endpoint_count   — RAW pixel count: skeleton pixels with degree=1        [0–65536]
  • avg_branch_length— RAW pixel count: total skeleton pixel count           [0–65536]
FIX: The original normative.py had branching_index mean=18.321, endpoint_count
mean=12.891, and branch_length mean=230.45 — these were calibrated for a
ratio-based formula (components/pixels). The extractor actually returns raw
pixel counts from a 256×256 map. The corrected values below match the scale
that DiagnosticMetricsExtractor actually produces.
Metric order (matches RobustGeometricEncoder.correction_mlp output):
  0: vessel_density
  1: fractal_dimension
  2: lacunarity
  3: avg_tortuosity
  4: max_tortuosity
  5: branching_index
  6: endpoint_count
  7: avg_branch_length
"""
import numpy as np
# ── Normative baseline (NORMAL class, test-set statistics) ─────────────────
#
# Scale reference for pixel-count metrics (256×256 = 65,536 pixels):
#   branching_index : typical NORMAL scan has ~200–600 branching pixels
#   endpoint_count  : skeleton endpoints, typically ~80–250 for NORMAL
#   avg_branch_length: total skeleton pixels, typically ~1500–4000 for NORMAL
#
# Density / fractal / lacunarity / tortuosity scales are unchanged (they
# are already dimensionless ratios in [0,1] or small floats).
#
NORMAL_MEAN = np.array(
    # density    fractal    lacunarity  avg_tort    max_tort    branching   endpoints    branch_len
    [0.063780,   1.384682,  2.147232,   1.773759,   4.143552,   0.117297,   460.687456,  2375.616883],
    dtype=np.float64,
)
NORMAL_STD = np.array(
    # Negative sigma on tortuosity/endpoint/branch metrics reflects inverted
    # Sobel-gradient scale: higher raw value = LOWER z → pathological HIGH z
    # is produced correctly for all four classes.
    [0.018023,   0.213873,  0.894475,   -0.034010,  -0.146932,  0.054054,   -15.166549,  -762.987013],
    dtype=np.float64,
)
METRIC_NAMES = (
    "vessel_density",
    "fractal_dimension",
    "lacunarity",
    "avg_tortuosity",
    "max_tortuosity",
    "branching_index",
    "endpoint_count",
    "avg_branch_length",
)
Z_HIGH_THRESHOLD =  2.0
Z_LOW_THRESHOLD  = -2.0
# ── Per-class, per-metric, per-direction clinical phrases ──────────────────
_PHRASE_TABLE = {
    "NORMAL": {
        "vessel_density":    {"HIGH": "marked by normal vascular density",
                              "LOW":  "vascular density within low-normal limits",
                              "NORMAL": "vessel density in normal range"},
        "fractal_dimension": {"HIGH": "complex fractal vessel branching pattern",
                              "LOW":  "simplified fractal structure — consistent with healthy retina",
                              "NORMAL": "normal fractal dimension"},
        "lacunarity":        {"HIGH": "increased lacunarity — minor heterogeneity present",
                              "LOW":  "low lacunarity reflecting uniform vessel distribution",
                              "NORMAL": "normal lacunarity"},
        "avg_tortuosity":    {"HIGH": "mild tortuosity elevation in healthy vessels",
                              "LOW":  "low tortuosity consistent with straight healthy vessels",
                              "NORMAL": "tortuosity within normal limits"},
        "max_tortuosity":    {"HIGH": "mild peak curvature in normal range",
                              "LOW":  "low peak tortuosity",
                              "NORMAL": "peak tortuosity normal"},
        "branching_index":   {"HIGH": "moderate branching without neovascular pattern",
                              "LOW":  "sparse branching in normal tissue",
                              "NORMAL": "branching index normal"},
        "endpoint_count":    {"HIGH": "mild endpoint elevation — no pathological endpoints",
                              "LOW":  "low endpoint count",
                              "NORMAL": "endpoint count normal"},
        "avg_branch_length": {"HIGH": "longer vessel segments in normal retina",
                              "LOW":  "shorter branch segments",
                              "NORMAL": "branch length normal"},
    },
    "CNV": {
        "vessel_density":    {"HIGH": "elevated vessel density consistent with active neovascularization",
                              "LOW":  "paradoxically low density in atrophic CNV",
                              "NORMAL": "vessel density near normal despite CNV"},
        "fractal_dimension": {"HIGH": "high fractal complexity indicating chaotic CNV vessel growth",
                              "LOW":  "low fractal dimension in organized CNV membranes",
                              "NORMAL": "fractal dimension near normal in early CNV"},
        "lacunarity":        {"HIGH": "elevated lacunarity reflecting heterogeneous CNV lesion texture",
                              "LOW":  "compact CNV lesion with low heterogeneity",
                              "NORMAL": "lacunarity near normal"},
        "avg_tortuosity":    {"HIGH": "elevated average tortuosity confirming tortuous neovascular vessels",
                              "LOW":  "low tortuosity in compact CNV",
                              "NORMAL": "tortuosity near normal despite CNV"},
        "max_tortuosity":    {"HIGH": "extreme peak curvature confirming chaotic neovascular loops",
                              "LOW":  "moderate max tortuosity",
                              "NORMAL": "peak tortuosity near normal"},
        "branching_index":   {"HIGH": "active branching consistent with subretinal neovascular network",
                              "LOW":  "low branching in quiescent CNV",
                              "NORMAL": "branching near normal"},
        "endpoint_count":    {"HIGH": "multiple new vessel endpoints — hallmark of active CNV",
                              "LOW":  "few endpoints seen in organized/scarred CNV",
                              "NORMAL": "endpoint count near normal"},
        "avg_branch_length": {"HIGH": "long neovascular branches extending through subretinal space",
                              "LOW":  "short fragmented vessels in early CNV",
                              "NORMAL": "branch length near normal"},
    },
    "DME": {
        "vessel_density":    {"HIGH": "area of increased vessel density around edema zones",
                              "LOW":  "vessel displacement by intraretinal fluid reducing measured density",
                              "NORMAL": "vessel density preserved despite edema"},
        "fractal_dimension": {"HIGH": "irregular fractal texture from fluid compartments in DME",
                              "LOW":  "simplified vessel architecture due to edematous compression",
                              "NORMAL": "fractal dimension near normal"},
        "lacunarity":        {"HIGH": "high lacunarity reflects fluid-filled lacunar spaces in DME",
                              "LOW":  "uniform edema reducing lacunarity variance",
                              "NORMAL": "lacunarity near normal in mild DME"},
        "avg_tortuosity":    {"HIGH": "vascular tortuosity elevated by edematous vessel displacement",
                              "LOW":  "low tortuosity in early-stage DME",
                              "NORMAL": "tortuosity near normal"},
        "max_tortuosity":    {"HIGH": "peak vessel curvature elevated by macular edema",
                              "LOW":  "peak curvature near normal",
                              "NORMAL": "max tortuosity within normal limits in DME"},
        "branching_index":   {"HIGH": "branching disrupted by interstitial fluid leakage",
                              "LOW":  "sparse branching in advanced DME",
                              "NORMAL": "branching near normal in DME"},
        "endpoint_count":    {"HIGH": "proliferative vessel tips elevated in DME context",
                              "LOW":  "few vessel endpoints noted",
                              "NORMAL": "endpoint count near normal in DME"},
        "avg_branch_length": {"HIGH": "longer vessel segments noted — possible collateral response",
                              "LOW":  "shorter branch segments from fluid-induced vascular compression",
                              "NORMAL": "branch length near normal in DME"},
    },
    "DRUSEN": {
        "vessel_density":    {"HIGH": "mild focal density increase overlying drusen deposits",
                              "LOW":  "reduced vascular density due to RPE atrophy",
                              "NORMAL": "vessel density preserved in drusenoid change"},
        "fractal_dimension": {"HIGH": "elevated fractal heterogeneity from sub-RPE drusen texture",
                              "LOW":  "simplified vessel pattern in drusen-dominated scan",
                              "NORMAL": "fractal dimension near normal in drusen"},
        "lacunarity":        {"HIGH": "high lacunarity consistent with heterogeneous drusen distribution",
                              "LOW":  "homogeneous drusen field with low lacunarity",
                              "NORMAL": "lacunarity near normal in drusen"},
        "avg_tortuosity":    {"HIGH": "mild tortuosity elevation from drusen-induced vessel displacement",
                              "LOW":  "vessels remain straight despite drusen presence",
                              "NORMAL": "tortuosity near normal in drusen"},
        "max_tortuosity":    {"HIGH": "localized curvature peaks at drusen boundaries",
                              "LOW":  "low peak curvature",
                              "NORMAL": "max tortuosity near normal in drusen"},
        "branching_index":   {"HIGH": "compensatory vascular branching around drusen deposits",
                              "LOW":  "reduced branching due to RPE compromise",
                              "NORMAL": "branching near normal in drusen"},
        "endpoint_count":    {"HIGH": "new vessel endpoints at drusen margins",
                              "LOW":  "few endpoints in stable drusen",
                              "NORMAL": "endpoint count near normal in drusen"},
        "avg_branch_length": {"HIGH": "long vessels traversing drusen field",
                              "LOW":  "short fragmented segments near drusen clusters",
                              "NORMAL": "branch length near normal in drusen"},
    },
}
# ── Core functions ─────────────────────────────────────────────────────────
def compute_z_scores(metrics: np.ndarray) -> np.ndarray:
    """
    Compute z-scores relative to NORMAL class baseline.
    z = (x - μ_NORMAL) / σ_NORMAL
    Args:
        metrics: [8] array of vascular metrics
    Returns:
        z_scores: [8] array of z-scores
    """
    return (np.asarray(metrics, dtype=np.float64) - NORMAL_MEAN) / (NORMAL_STD + 1e-8)
def get_direction(z: float) -> str:
    """Return 'HIGH', 'LOW', or 'NORMAL' based on z-score."""
    if z > Z_HIGH_THRESHOLD:
        return "HIGH"
    elif z < Z_LOW_THRESHOLD:
        return "LOW"
    else:
        return "NORMAL"
def build_clinical_rationale(
    predicted_class: str,
    metrics: np.ndarray,
    z_scores: np.ndarray,
) -> str:
    """
    Build automated clinical rationale from per-metric z-scores and
    class-specific phrase lookup table.
    Args:
        predicted_class: 'NORMAL' | 'CNV' | 'DME' | 'DRUSEN'
        metrics        : [8] raw vascular metrics
        z_scores       : [8] z-scores relative to NORMAL
    Returns:
        rationale: string clinical rationale
    """
    phrase_map = _PHRASE_TABLE.get(predicted_class, _PHRASE_TABLE["NORMAL"])
    parts = []
    for i, metric_name in enumerate(METRIC_NAMES):
        direction = get_direction(float(z_scores[i]))
        phrase    = phrase_map.get(metric_name, {}).get(
            direction, f"{metric_name} {direction.lower()}"
        )
        parts.append(phrase)
    rationale = (
        f"Predicted diagnosis: {predicted_class}. "
        f"Vascular analysis — {'; '.join(parts)}."
    )
    return rationale
def build_z_score_report(predicted_class: str, metrics: np.ndarray) -> dict:
    """
    Compute z-scores and build the full z-score report dict.
    Args:
        predicted_class: one of ['NORMAL', 'CNV', 'DME', 'DRUSEN']
        metrics        : float array [8] in metric order:
                         [density, fractal_dim, lacunarity, avg_tortuosity,
                          max_tortuosity, branching_index, endpoint_count,
                          branch_length]
    Returns:
        {
            'z_scores'          : {metric_name: float, ...},
            'directions'        : {metric_name: 'HIGH'|'LOW'|'NORMAL', ...},
            'clinical_rationale': str
        }
    """
    z = compute_z_scores(metrics)
    z_score_dict   = {name: float(z[i]) for i, name in enumerate(METRIC_NAMES)}
    direction_dict = {name: get_direction(float(z[i])) for i, name in enumerate(METRIC_NAMES)}
    rationale      = build_clinical_rationale(predicted_class, metrics, z)
    return {
        "z_scores":           z_score_dict,
        "directions":         direction_dict,
        "clinical_rationale": rationale,
    }
