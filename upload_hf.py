"""Upload AEGIS Security Environment to HuggingFace Hub."""
import os
from huggingface_hub import HfApi, create_repo

REPO_ID = "Hagenbefragen/openenv-aegis-security"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

api = HfApi()

# Create repo (Space with static HTML)
try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
    )
    print(f"Created HF Space: {REPO_ID}")
except Exception as e:
    print(f"Repo creation: {e}")

# Upload all files
files_to_upload = [
    ("index.html", "index.html"),
    ("README.md", "README.md"),
    ("aegis_env.py", "aegis_env.py"),
    ("cascade_sim.py", "cascade_sim.py"),
    ("models.py", "models.py"),
    ("server/app.py", "server/app.py"),
    ("Dockerfile", "Dockerfile"),
    ("pyproject.toml", "pyproject.toml"),
    ("HACKATHON_SUBMISSION.md", "HACKATHON_SUBMISSION.md"),
]

for local_path, repo_path in files_to_upload:
    full_path = os.path.join(LOCAL_DIR, local_path)
    if os.path.exists(full_path):
        try:
            api.upload_file(
                path_or_fileobj=full_path,
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="space",
            )
            print(f"  Uploaded: {repo_path}")
        except Exception as e:
            print(f"  Error uploading {repo_path}: {e}")

print(f"\nDone! View at: https://huggingface.co/spaces/{REPO_ID}")
