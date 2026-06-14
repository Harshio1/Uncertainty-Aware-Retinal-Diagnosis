from huggingface_hub import HfApi
import os

os.environ["HF_TOKEN"] = "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"
api = HfApi()
info = api.space_info(repo_id="Harshio/adaptive-oct-classifier")
print(f"Space Status: {info.runtime.stage}")
