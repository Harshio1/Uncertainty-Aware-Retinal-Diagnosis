import os
import torch
from inference import ModelEngine

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weights",
    "final_patent_architecture.pth",
)

print(f"[*] Testing model load from {WEIGHTS_PATH}")
try:
    engine = ModelEngine(weights_path=WEIGHTS_PATH)
    print("[*] Model loaded successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[!] Error: {e}")
