from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path=".",
    repo_id="Harshio/adaptive-oct-classifier",
    repo_type="space",
    ignore_patterns=[
        "venv/*", 
        "__pycache__/*", 
        "tmp.txt", 
        ".git/*",
        ".github/*",
        "push_out.txt",
        "hf_status.json",
        "check_space.py",
        "check_status.py"
    ]
)
