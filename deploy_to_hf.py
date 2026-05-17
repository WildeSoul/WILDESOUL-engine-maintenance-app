"""
Hosting Script: deploy_to_hf.py
Deploys the trained model and Streamlit app to HuggingFace Spaces.
Can be run locally or as part of CI/CD pipeline.
"""
import os
import sys
import glob
import argparse
from huggingface_hub import HfApi, Repository

def deploy(hf_token=None, space_id="WILDESOUL/engine-maintenance-app"):
    """Deploy model and app to HuggingFace Space."""
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found. Set it via --token or HF_TOKEN env var.")
        sys.exit(1)

    api = HfApi()
    user_info = api.whoami(token=token)
    print(f"Authenticated as: {user_info['name']}")

    # ── Step 1: Upload model artifacts ──
    model_repo = f"{user_info['name']}/engine-maintenance-model"
    api.create_repo(repo_id=model_repo, exist_ok=True, token=token)

    model_files = {
        "model_building/best_model.joblib": "best_model.joblib",
        "model_building/feature_info.json": "feature_info.json",
        "model_building/model_comparison.json": "model_comparison.json",
    }
    for local, remote in model_files.items():
        if os.path.exists(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                           repo_id=model_repo, token=token)
            print(f"  ✅ Uploaded {local} → {model_repo}/{remote}")

    # Upload plots
    for fpath in glob.glob("model_building/plots/*.png"):
        api.upload_file(path_or_fileobj=fpath,
                       path_in_repo=f"plots/{os.path.basename(fpath)}",
                       repo_id=model_repo, token=token)
        print(f"  ✅ Uploaded {fpath}")

    print(f"\n🏆 Model registry: https://huggingface.co/{model_repo}")

    # ── Step 2: Deploy Streamlit app ──
    api.create_repo(repo_id=space_id, repo_type="space",
                    space_sdk="streamlit", exist_ok=True, token=token)

    deploy_files = {
        "deployment/app.py": "app.py",
        "deployment/requirements.txt": "requirements.txt",
        "deployment/README.md": "README.md",
        "model_building/best_model.joblib": "best_model.joblib",
        "model_building/feature_info.json": "feature_info.json",
        "model_building/model_comparison.json": "model_comparison.json",
    }
    for local, remote in deploy_files.items():
        if os.path.exists(local):
            api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                           repo_id=space_id, repo_type="space", token=token)
            print(f"  ✅ Deployed {local} → {space_id}/{remote}")

    print(f"\n🚀 App live at: https://huggingface.co/spaces/{space_id}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to HuggingFace")
    parser.add_argument("--token", help="HuggingFace API token", default=None)
    parser.add_argument("--space", help="HF Space ID", default="WILDESOUL/engine-maintenance-app")
    args = parser.parse_args()
    deploy(hf_token=args.token, space_id=args.space)
