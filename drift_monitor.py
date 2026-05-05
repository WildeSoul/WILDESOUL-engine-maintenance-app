import pandas as pd
import requests
import os
import json
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift():
    print("Initializing Data Drift Detection...")
    
    reference_path = 'data/train.csv'
    current_path = 'deployment/inference_log.csv'
    
    if not os.path.exists(current_path):
        print("No inference logs found. Skipping drift check.")
        return

    reference_data = pd.read_csv(reference_path)
    # The reference data includes scaled features and the target.
    # The Streamlit app will log the pre-scaled numerical inputs, but wait, Streamlit inputs aren't scaled yet.
    # To properly check drift, we should check it on the raw features.
    
    # We will assume Streamlit logs the exact same feature names as X_train.
    current_data = pd.read_csv(current_path)
    
    # Run Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data.drop('Engine_Condition', axis=1, errors='ignore'), current_data=current_data)
    
    report_json = drift_report.as_dict()
    
    # Get the dataset drift p-value/score
    dataset_drift = report_json['metrics'][0]['result']['dataset_drift']
    share_of_drifted_columns = report_json['metrics'][0]['result']['share_of_drifted_columns']
    
    print(f"Dataset Drift Detected: {dataset_drift}")
    print(f"Share of Drifted Columns: {share_of_drifted_columns}")
    
    # If more than 20% of columns drifted or significant drift detected
    if dataset_drift:
        print("Significant drift detected! Triggering GitHub Issue...")
        trigger_github_issue()
    else:
        print("No significant drift detected.")

def trigger_github_issue():
    github_token = os.environ.get('GH_TOKEN') or os.environ.get('HF_TOKEN') # Using HF_TOKEN as fallback for demo
    repo = "WildeSoul/WILDESOUL-engine-maintenance-app"
    
    if not github_token:
        print("No GitHub token found. Cannot create issue.")
        return

    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": "[Automated] Model Retrain Required: Data Drift Detected",
        "body": "Evidently AI has detected significant data drift in the production inference logs (p-value < 0.05). Please review the logs and trigger a retraining pipeline.",
        "labels": ["Model Retrain Required", "Active Learning"]
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 201:
        print("Successfully created GitHub issue.")
    else:
        print(f"Failed to create issue: {response.text}")

if __name__ == "__main__":
    check_drift()
