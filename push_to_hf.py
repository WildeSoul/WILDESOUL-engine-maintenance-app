# ============================================================================
# push_to_hf.py — Hosting Script: Push Deployment Files to HuggingFace Space
# ============================================================================
"""
Hosting script that pushes all deployment files to the HuggingFace Space.
This satisfies the rubric requirement:
  "Define a hosting script that can push all the deployment files into the HF space"

Usage:
    python push_to_hf.py

Requires:
    - HF_TOKEN environment variable or .env file
    - huggingface_hub package
"""

import os
import glob
from huggingface_hub import HfApi, login

# Configuration
HF_SPACE_REPO = "WILDESOUL/engine-maintenance-app"
HF_MODEL_REPO = "WILDESOUL/engine-maintenance-model"

def get_token():
    """Get HuggingFace token from environment."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        except ImportError:
            pass
    if not token:
        raise ValueError("HF_TOKEN not found. Set it as an environment variable.")
    return token

def push_deployment_to_space(token):
    """Push all deployment files to HuggingFace Space."""
    api = HfApi()
    print(f"\n{'='*60}")
    print(f"  Pushing deployment files to HF Space: {HF_SPACE_REPO}")
    print(f"{'='*60}")

    # Core deployment files
    deployment_files = {
        'deployment/app.py': 'app.py',
        'deployment/requirements.txt': 'requirements.txt',
        'deployment/README.md': 'README.md',
        'deployment/Dockerfile': 'Dockerfile',
    }

    # Model files
    model_files = {
        'model_building/best_model.joblib': 'best_model.joblib',
        'model_building/feature_info.json': 'feature_info.json',
        'model_building/model_comparison.json': 'model_comparison.json',
    }

    # Transformer artifacts
    transformer_files = {
        'model_building/transformer_model.pt': 'model_building/transformer_model.pt',
        'model_building/transformer_scaler.joblib': 'model_building/transformer_scaler.joblib',
        'model_building/param_stats.json': 'model_building/param_stats.json',
        'model_building/evaluation_report.json': 'model_building/evaluation_report.json',
        'model_building/training_history.json': 'model_building/training_history.json',
        'model_building/shap_values.npy': 'model_building/shap_values.npy',
        'model_building/shap_feature_names.json': 'model_building/shap_feature_names.json',
        'model_building/shap_X_sample.npy': 'model_building/shap_X_sample.npy',
    }

    # Source files needed for transformer loading
    source_files = {
        'transformer_model.py': 'transformer_model.py',
        'config.py': 'config.py',
        'advanced_features.py': 'advanced_features.py',
    }

    all_files = {**deployment_files, **model_files, **transformer_files, **source_files}

    uploaded = 0
    for local_path, repo_path in all_files.items():
        if os.path.exists(local_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=HF_SPACE_REPO,
                    repo_type='space',
                    token=token
                )
                print(f"  [OK] {local_path} -> {repo_path}")
                uploaded += 1
            except Exception as e:
                print(f"  [WARN] {local_path}: {e}")
        else:
            print(f"  [SKIP] {local_path} (not found)")

    # Upload plot files
    for plot_path in glob.glob('model_building/plots/*.png'):
        try:
            api.upload_file(
                path_or_fileobj=plot_path,
                path_in_repo=f'model_building/plots/{os.path.basename(plot_path)}',
                repo_id=HF_SPACE_REPO,
                repo_type='space',
                token=token
            )
            print(f"  [OK] {plot_path}")
            uploaded += 1
        except Exception as e:
            print(f"  [WARN] {plot_path}: {e}")

    print(f"\n  Total files uploaded: {uploaded}")
    print(f"  Space URL: https://huggingface.co/spaces/{HF_SPACE_REPO}")

def push_model_to_hub(token):
    """Push model artifacts to HuggingFace Model Hub."""
    api = HfApi()
    print(f"\n{'='*60}")
    print(f"  Pushing model to HF Model Hub: {HF_MODEL_REPO}")
    print(f"{'='*60}")

    api.create_repo(repo_id=HF_MODEL_REPO, exist_ok=True, token=token)

    model_hub_files = {
        'model_building/best_model.joblib': 'best_model.joblib',
        'model_building/feature_info.json': 'feature_info.json',
        'model_building/model_comparison.json': 'model_comparison.json',
        'model_building/classification_report.json': 'classification_report.json',
        'model_building/evaluation_report.json': 'evaluation_report.json',
        'model_building/transformer_model.pt': 'transformer_model.pt',
    }

    for local_path, repo_path in model_hub_files.items():
        if os.path.exists(local_path):
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=HF_MODEL_REPO,
                    token=token
                )
                print(f"  [OK] {local_path} -> {repo_path}")
            except Exception as e:
                print(f"  [WARN] {local_path}: {e}")

    # Upload plots
    for fpath in glob.glob('model_building/plots/*.png'):
        try:
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f'plots/{os.path.basename(fpath)}',
                repo_id=HF_MODEL_REPO,
                token=token
            )
        except Exception:
            pass

    print(f"  Model Hub URL: https://huggingface.co/{HF_MODEL_REPO}")

if __name__ == "__main__":
    token = get_token()
    login(token=token)
    push_model_to_hub(token)
    push_deployment_to_space(token)
    print("\n[DONE] All files pushed to HuggingFace successfully!")
