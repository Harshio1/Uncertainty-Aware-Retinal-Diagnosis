import sys, re

with open('cam_v6.py', 'r', encoding='utf-8') as f:
    text1 = f.read()

with open('xai_v6.py', 'r', encoding='utf-8') as f:
    text2 = f.read()
    
xai_imports = """\"\"\"
xai.py — PUBLICATION-GRADE XAI SUITE (Exact Port of Cell 16)
\"\"\"
import os, gc, glob, math, json, time, warnings, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import base64
import io

try:
    from sklearn.metrics import auc
except ImportError:
    pass

from normative import NORMAL_MEAN as mean_normal
from normative import NORMAL_STD as std_normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD  = np.array([0.229, 0.224, 0.225])

"""

xai_gradio = """
# ─────────────────────────────────────────────────────────────────────────────
# WRAPPER FOR GRADIO API
# ─────────────────────────────────────────────────────────────────────────────
def run_gradcam_pipeline(model: torch.nn.Module, img_vis: torch.Tensor, img_geo: torch.Tensor, target_class: int, use_unet: bool = True) -> str:
    \"\"\"
    Wrapper for FastAPI/Gradio app to use Cell 16's exact backend compute_saliency.
    \"\"\"
    gcam, cafm_norm, combined, geo_224, img_np = compute_saliency(model, img_vis, img_geo, target_class)
    
    cam_uint8 = np.uint8(255 * combined)
    _, mask = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap[mask == 0] = 0
    
    orig_img_uint8 = np.uint8(255 * img_np)
    orig_img_uint8 = cv2.cvtColor(orig_img_uint8, cv2.COLOR_RGB2BGR)
    
    overlay_alpha = 0.55
    overlay = cv2.addWeighted(heatmap, overlay_alpha, orig_img_uint8, 1 - overlay_alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    img_pil = Image.fromarray(overlay_rgb)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return b64_str
"""

cam_m = re.search(r'(class NativeGradCAM.*?remove_hooks.*?self\.handles = \[\]\n)', text1, re.DOTALL)
funcs_m = re.search(r'(def compute_saliency.*?)(?=\n# ── 7\. |\Z)', text2, re.DOTALL)

if cam_m and funcs_m:
    with open('xai.py', 'w', encoding='utf-8') as f:
        f.write(xai_imports)
        f.write(cam_m.group(1).strip() + '\n\n')
        f.write(funcs_m.group(1).strip() + '\n')
        f.write(xai_gradio)
    print('xai.py generated successfully')
else:
    print('failed to extract patterns')
