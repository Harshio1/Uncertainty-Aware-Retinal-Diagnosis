"""
main.py — FastAPI server for Adaptive OCT Classification API.

Patent-aligned retinal OCT classification system using:
  Bidirectional CAFM + EfficientNetV2-S + LightweightUNet
  MC Dropout (T=20), temperature scaling, Grad-CAM with Otsu thresholding,
  and z-score vascular analysis.

Reconstructed from compiled bytecode.
"""

import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from inference import ModelEngine, CLASSES
from normative import METRIC_NAMES, NORMAL_MEAN, NORMAL_STD, build_z_score_report
from preprocessing import preprocess_image
from xai import run_gradcam_pipeline

# ── Config ─────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights",
    "final_patent_architecture.pth",
)
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TEMPERATURE = float(os.environ.get("TEMP_SCALE", "1.0"))

engine: Optional[ModelEngine] = None

# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights on startup; release on shutdown."""
    global engine
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Weights: {WEIGHTS_PATH}")
    print(f"[*] Temperature: {TEMPERATURE}")

    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(
            f"Weight file not found: {WEIGHTS_PATH}\n"
            "Place final_patent_architecture.pth in weights/ directory."
        )

    engine = ModelEngine(weights_path=WEIGHTS_PATH)
    print("[*] Model engine ready. Server accepting requests.")
    yield
    print("[*] Shutting down model engine.")
    engine = None


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive OCT Classification API",
    description=(
        "Patent-aligned retinal OCT classification system using Bidirectional CAFM "
        "+ EfficientNetV2-S + LightweightUNet. Implements MC Dropout (T=20), "
        "temperature scaling, Grad-CAM with Otsu thresholding, and z-score vascular "
        "analysis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Internal helper ────────────────────────────────────────────────────────

def build_response(
    img_vis,
    img_geo,
    image_bytes: bytes,
    inference_time_ms: float,
    predicted_class: str,
    confidence: float,
    uncertainty: float,
    class_probabilities: dict,
    vascular_metrics: dict,
) -> dict:
    """Build the full clinical JSON response dict."""

    raw_metrics_np = np.array(
        [
            vascular_metrics.get("density", 0.0),
            vascular_metrics.get("fractal_dim", 0.0),
            vascular_metrics.get("lacunarity", 0.0),
            vascular_metrics.get("avg_tortuosity", 0.0),
            vascular_metrics.get("max_tortuosity", 0.0),
            vascular_metrics.get("branching_index", 0.0),
            vascular_metrics.get("endpoint_count", 0.0),
            vascular_metrics.get("branch_length", 0.0),
        ],
        dtype=np.float64,
    )

    z_report = build_z_score_report(predicted_class, raw_metrics_np)

    # Grad-CAM
    saliency_b64 = ""
    try:
        target_idx   = CLASSES.index(predicted_class)
        saliency_b64 = run_gradcam_pipeline(
            model=engine.model,
            img_vis=img_vis,
            img_geo=img_geo,
            target_class=target_idx,
            use_unet=True,
        )
    except Exception as exc:
        print(f"[!] GradCAM failed: {exc}")

    if confidence >= 0.90:
        status_text = "HIGH CONFIDENCE"
    elif confidence >= 0.70:
        status_text = "MODERATE CONFIDENCE"
    else:
        status_text = "LOW CONFIDENCE"

    z_scores_formatted = {
        metric: {"z": round(z_val, 2), "flag": z_report["directions"][metric]}
        for metric, z_val in z_report["z_scores"].items()
    }

    class_probs_formatted = {
        cls: {"probability": f"{class_probabilities.get(cls, 0.0) * 100:.2f}%"}
        for cls in CLASSES
    }

    vascular_biomarkers = {
        "density":           round(vascular_metrics.get("density", 0.0), 2),
        "fractal_dimension": round(vascular_metrics.get("fractal_dim", 0.0), 2),
        "lacunarity":        round(vascular_metrics.get("lacunarity", 0.0), 2),
        "avg_tortuosity":    round(vascular_metrics.get("avg_tortuosity", 0.0), 2),
        "max_tortuosity":    round(vascular_metrics.get("max_tortuosity", 0.0), 2),
        "branching_index":   int(round(vascular_metrics.get("branching_index", 0.0))),
        "endpoint_count":    int(round(vascular_metrics.get("endpoint_count", 0.0))),
        "branch_length":     int(round(vascular_metrics.get("branch_length", 0.0))),
    }

    return {
        "model_prediction": {
            "diagnosis":             predicted_class,
            "calibrated_confidence": f"{confidence * 100:.2f}%",
            "epistemic_uncertainty": f"±{uncertainty * 100:.2f}%",
            "status":                status_text,
        },
        "class_probabilities":  class_probs_formatted,
        "calibration_info":     {"temperature": TEMPERATURE, "mc_dropout_passes": 20},
        "vascular_biomarkers":  vascular_biomarkers,
        "z_scores":             z_scores_formatted,
        "clinical_rationale":   z_report["clinical_rationale"],
        "saliency_image_b64":   saliency_b64,
        "inference_time_ms":    round(inference_time_ms, 1),
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"title": "Adaptive OCT Classification API", "version": "1.0.0", "status": "online"}


@app.get("/health")
def health():
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status":     "healthy",
        "model":      "FinalPatentArchitecture",
        "backbone":   "tf_efficientnetv2_s.in21k_ft_in1k",
        "device":     DEVICE,
        "temperature": TEMPERATURE,
    }


@app.get("/classes")
def get_classes():
    return {
        "Model": {
            "NORMAL": "Healthy retina — no pathology detected",
            "CNV":    "Choroidal Neovascularization",
            "DME":    "Diabetic Macular Edema",
            "DRUSEN": "Drusen deposits — AMD precursor",
        }
    }


@app.get("/metrics/normative")
def get_normative():
    """Return μ and σ for NORMAL class metrics used in z-score computation."""
    return {
        "formula":     "z = (x - mu_NORMAL) / sigma_NORMAL",
        "source":      "Kermany2018 OCT Dataset — NORMAL class test split",
        "description": (
            "These baseline statistics are used to compute per-metric Z-scores "
            "for pathological deviation quantification relative to healthy controls."
        ),
        "metrics": {
            name: {
                "mu_NORMAL":    float(NORMAL_MEAN[i]),
                "sigma_NORMAL": float(NORMAL_STD[i]),
            }
            for i, name in enumerate(METRIC_NAMES)
        },
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Single OCT image inference.

    - Preprocessing: CLAHE (8x8, clip=2.0) → visual (224x224 ImageNet) + geo (512x512 [0,1])
    - Inference: MC Dropout T=20, temperature scaling
    - XAI: Grad-CAM + Otsu + JET overlay (α=0.55) as base64 PNG
    - Z-scores: deviation from NORMAL class baseline

    Returns full clinical JSON.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Got: {file.content_type}",
        )

    image_bytes = await file.read()

    try:
        img_vis, img_geo = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Image decode error: {exc}")

    try:
        t0     = time.perf_counter()
        result = engine.predict_single(img_vis, img_geo)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        response = build_response(
            img_vis=img_vis,
            img_geo=img_geo,
            image_bytes=image_bytes,
            inference_time_ms=elapsed_ms,
            predicted_class=result["prediction"],
            confidence=result["confidence"],
            uncertainty=result["uncertainty"],
            class_probabilities=result["all_probabilities"],
            vascular_metrics=result["vascular_metrics"],
        )
        return JSONResponse(content=response)

    except Exception as exc:
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}\n{tb}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Batch OCT image inference (max 16 images).
    Returns a list of clinical JSON objects, one per image.
    """
    MAX_BATCH = 16
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(files)} exceeds maximum {MAX_BATCH}",
        )

    responses = []
    for file in files:
        if not file.content_type.startswith("image/"):
            responses.append(
                {"error": f"File '{file.filename}' is not an image (got {file.content_type})."}
            )
            continue
        image_bytes = await file.read()
        try:
            img_vis, img_geo = preprocess_image(image_bytes)
            result = engine.predict_single(img_vis, img_geo)
            response = build_response(
                img_vis=img_vis,
                img_geo=img_geo,
                image_bytes=image_bytes,
                inference_time_ms=0.0,
                predicted_class=result["prediction"],
                confidence=result["confidence"],
                uncertainty=result["uncertainty"],
                class_probabilities=result["all_probabilities"],
                vascular_metrics=result["vascular_metrics"],
            )
            responses.append(response)
        except Exception as exc:
            responses.append({"error": str(exc)})

    return JSONResponse(content=responses)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
