print("Checking imports...")
try:
    import torch
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    import gradio
    print(f"gradio version: {gradio.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")
