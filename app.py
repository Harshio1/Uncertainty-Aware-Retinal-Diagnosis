"""
app.py — Refactored Gradio application for Adaptive OCT Inference.
Theme: Clean Dark Minimal (Professional ML Tool Style)
"""

import os
import io
import json
import base64
import traceback

import gradio as gr
from PIL import Image
import numpy as np

from inference import ModelEngine, CLASSES
from preprocessing import preprocess_image
from xai import run_gradcam_pipeline
from normative import build_z_score_report

# ── Config ─────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights",
    "final_patent_architecture.pth",
)
TEMPERATURE = 1.4060

# ── Lazy model loader ──────────────────────────────────────────────────────
_engine: ModelEngine | None = None


def load_model() -> ModelEngine:
    global _engine
    if _engine is None:
        print("[*] Loading model for Gradio app …")
        _engine = ModelEngine(weights_path=WEIGHTS_PATH)
        print("[*] Model ready.")
    return _engine


# ── LLM clinical rationale ─────────────────────────────────────────────────

def _llm_rationale(json_report: dict, saliency_b64: str) -> str:
    api_key = os.environ.get("HF_TOKEN", "").strip()

    if not api_key:
        return json_report.get("clinical_rationale", "")

    try:
        from huggingface_hub import InferenceClient

        report_clean = {
            k: v for k, v in json_report.items()
            if k not in {"saliency_image_b64"}
        }

        messages = [
            {
                "role": "user",
                "content": (
                    "You are a senior ophthalmologist and retinal imaging AI specialist. "
                    "Analyze the following OCT scan AI classification report.\n\n"
                    "Write a concise, clinically accurate rationale (2-4 sentences) that:\n"
                    "  1. Confirms or contextualises the predicted diagnosis.\n"
                    "  2. References the most significant vascular biomarker deviations.\n\n"
                    "IMPORTANT: Output ONLY the rationale text itself. "
                    "Do NOT include any headers. Start directly with the clinical findings.\n\n"
                    f"Report JSON:\n{json.dumps(report_clean, indent=2)}"
                ),
            }
        ]

        client = InferenceClient(api_key=api_key)
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=messages,
            max_tokens=512,
        )
        rationale = response.choices[0].message.content.strip()

        import re
        noise_patterns = [
            r"^\**clinical rationale:?\**\s*",
            r"^\**rationale:?\**\s*",
            r"^\**note:?\**\s*",
            r"^the clinical rationale is:?\s*",
        ]
        for pattern in noise_patterns:
            rationale = re.sub(pattern, "", rationale, flags=re.IGNORECASE)

        return rationale.strip()

    except Exception:
        return json_report.get("clinical_rationale", "")


# ── Main inference callback ────────────────────────────────────────────────

def predict_oct(image: Image.Image):
    engine = load_model()

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    try:
        img_vis, img_geo = preprocess_image(image_bytes)
        result           = engine.predict_single(img_vis, img_geo)

        predicted_class  = result["prediction"]
        confidence       = result["confidence"]
        uncertainty      = result["uncertainty"]
        class_probs      = result["all_probabilities"]
        vascular_metrics = result["vascular_metrics"]

        status_text = "HIGH CONFIDENCE" if confidence >= 0.90 else "MODERATE CONFIDENCE" if confidence >= 0.70 else "LOW CONFIDENCE"

        raw_metrics_np = np.array(
            [
                vascular_metrics.get("density",         0.0),
                vascular_metrics.get("fractal_dim",     0.0),
                vascular_metrics.get("lacunarity",      0.0),
                vascular_metrics.get("avg_tortuosity",  0.0),
                vascular_metrics.get("max_tortuosity",  0.0),
                vascular_metrics.get("branching_index", 0.0),
                vascular_metrics.get("endpoint_count",  0.0),
                vascular_metrics.get("branch_length",   0.0),
            ],
            dtype=np.float64,
        )

        z_report = build_z_score_report(predicted_class, raw_metrics_np)

        z_scores_formatted = {
            metric: {"z": round(z_val, 2), "flag": z_report["directions"][metric]}
            for metric, z_val in z_report["z_scores"].items()
        }

        class_probs_formatted = {
            cls: {"probability": f"{class_probs.get(cls, 0.0) * 100:.2f}%"}
            for cls in CLASSES
        }

        vascular_biomarkers = {
            "density":           round(vascular_metrics.get("density",         0.0), 4),
            "fractal_dimension": round(vascular_metrics.get("fractal_dim",     0.0), 4),
            "lacunarity":        round(vascular_metrics.get("lacunarity",       0.0), 4),
            "avg_tortuosity":    round(vascular_metrics.get("avg_tortuosity",  0.0), 4),
            "max_tortuosity":    round(vascular_metrics.get("max_tortuosity",  0.0), 4),
            "branching_index":   round(vascular_metrics.get("branching_index", 0.0), 2),
            "endpoint_count":    round(vascular_metrics.get("endpoint_count",  0.0), 2),
            "branch_length":     round(vascular_metrics.get("branch_length",   0.0), 2),
        }

        saliency_b64 = ""
        saliency_img = image
        try:
            target_idx   = CLASSES.index(predicted_class)
            saliency_b64 = run_gradcam_pipeline(
                model=engine.model,
                img_vis=img_vis,
                img_geo=img_geo,
                target_class=target_idx,
                use_unet=True,
            )
            saliency_img = Image.open(io.BytesIO(base64.b64decode(saliency_b64)))
        except Exception: pass

        json_report = {
            "patient_scan_id": "OCT-TEST-001",
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
        }

        if predicted_class == "NORMAL" and confidence >= 0.85:
            json_report["clinical_rationale"] = "No pathological findings detected. Retinal layer architecture appears intact."
        else:
            json_report["clinical_rationale"] = _llm_rationale(json_report, saliency_b64)

        rationale_text = json_report.pop("clinical_rationale", "")
        return json_report, saliency_img, rationale_text

    except Exception:
        return {"error": traceback.format_exc()}, image, ""


# ── Custom CSS ─────────────────────────────────────────────────────────────

custom_css = """
/* Dark Theme Overrides with Background */
html, body {
    height: 100dvh !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
    background-image: url("file/assets/bg.png") !important;
    background-size: cover !important;
    background-position: center !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
    position: relative !important;
}

/* Semi-transparent dark blue overlay for readability */
body::before {
    content: "" !important;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background: rgba(2, 6, 23, 0.75) !important;
    z-index: -1 !important;
}

.gradio-container {
    background-color: transparent !important;
    backdrop-filter: blur(2px) !important;
    height: 100dvh !important;
    min-height: 100dvh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 1rem !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

.block {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

h1 {
    color: #f9fafb !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
}

label, .label-wrap span {
    color: #f3f4f6 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.025em !important;
    opacity: 1 !important;
    visibility: visible !important;
    z-index: 99 !important;
    position: relative !important;
}

/* Prominent Analyze Button */
button.primary {
    background-color: #2563eb !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.75rem !important;
    border-radius: 6px !important;
    transition: all 0.2s ease !important;
    margin-top: 10px !important;
}

button.primary:hover {
    background-color: #1d4ed8 !important;
    transform: translateY(-1px);
}

#main-row {
    flex: 1 !important;
    display: flex !important;
    align-items: stretch !important;
}

#left-col, #right-col {
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
}

#json-box {
    height: 400px !important;
    flex: 0 0 auto !important;
    overflow-y: auto !important;
}

#rationale-box {
    flex: 1 !important;
    margin-top: 0 !important;
}

#rationale-box textarea {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 16px !important;
    line-height: 1.6 !important;
    font-size: 1.05rem !important;
    color: #f3f4f6 !important;
}

#header-text, #sub-header-text {
    background: transparent !important;
    border: none !important;
    padding-left: 0 !important;
    margin-bottom: 0 !important;
}

#header-text h1 {
    font-size: 2.2rem !important;
    letter-spacing: -0.02em !important;
}

#sub-header-text p {
    color: #9ca3af !important;
    font-size: 1.1rem !important;
}

footer { display: none !important; }
"""

# ── Gradio UI ──────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Trust‑Aware OCT: A Bidirectional Cross‑Attention Model for Explainable Retinal Disease Diagnosis",
    theme=gr.themes.Default(
        primary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ),
    css=custom_css
) as demo:
    gr.Markdown("# Trust‑Aware OCT: A Bidirectional Cross‑Attention Model for Explainable Retinal Disease Diagnosis", elem_id="header-text")
    gr.Markdown("Upload an OCT retinal scan for comprehensive AI-driven diagnostic analysis.", elem_id="sub-header-text")

    with gr.Row(equal_height=True, elem_id="main-row"):
        with gr.Column(scale=1, elem_id="left-col"):
            input_image = gr.Image(type="pil", label="Patient OCT Scan", height=300, show_label=True)
            submit_btn  = gr.Button("Analyze Retinal Scan", variant="primary")
            output_saliency = gr.Image(
                type="pil",
                label="Diagnostic Attention Map (Grad-CAM)",
                interactive=False,
                height=300,
                show_label=True
            )
        with gr.Column(scale=1, elem_id="right-col"):
            output_json = gr.JSON(label="JSON Output", elem_id="json-box", show_label=True)
            output_rationale = gr.Textbox(
                label="Clinical Rationale (LLM)", 
                lines=10, 
                interactive=False,
                elem_id="rationale-box",
                show_label=True
            )

    submit_btn.click(
        fn=predict_oct,
        inputs=input_image,
        outputs=[output_json, output_saliency, output_rationale],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)