from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="model.py",
    path_in_repo="model.py",
    repo_id="Harshio/adaptive-oct-classifier",
    repo_type="space"
)
print("SUCCESS!")
