from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="requirements.txt",
    path_in_repo="requirements.txt",
    repo_id="Harshio/adaptive-oct-classifier",
    repo_type="space"
)
print("SUCCESS!")
