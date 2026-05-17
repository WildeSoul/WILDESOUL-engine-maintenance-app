# 🔧 Predictive Maintenance — Engine Health Monitoring

[![GitHub Actions](https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app/actions/workflows/pipeline.yml/badge.svg)](https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app/actions)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace-blue)](https://huggingface.co/spaces/WILDESOUL/engine-maintenance-app)

An end-to-end **MLOps pipeline** for predictive engine maintenance — from raw sensor data to a deployed Streamlit web app, fully automated via GitHub Actions.

---

## 📋 Problem Statement

Vehicle breakdowns lead to significant financial losses. This project builds a machine learning classification model that analyzes engine sensor data (RPM, temperature, pressure) to predict whether an engine requires maintenance — enabling proactive intervention before failure occurs.

## 🏗️ Project Structure

```
├── .github/workflows/
│   └── pipeline.yml              # CI/CD: Train → Evaluate → Deploy
├── data/
│   ├── engine_data.csv           # Raw dataset (19,536 rows)
│   ├── train.csv                 # Training split (80%)
│   └── test.csv                  # Test split (20%)
├── deployment/
│   ├── app.py                    # Streamlit web application
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # HF Space config
├── model_building/
│   ├── best_model.joblib         # Trained model pipeline
│   ├── feature_info.json         # Feature metadata
│   ├── model_comparison.json     # All model results
│   └── plots/                    # EDA & evaluation visualizations
├── predictive_maintenance.py     # Main ML script (jupytext)
├── predictive_maintenance.ipynb  # Jupyter Notebook
├── drift_monitor.py              # Data drift detection
├── feature_engineering.py        # Rolling window features
├── Interim_Report.md             # Interim business report
├── Final_Report.md               # Final business report
└── README.md                     # This file
```

## ⚙️ Pipeline Features

| Stage | Details |
|-------|---------|
| **Data Registration** | Hugging Face Datasets (`WILDESOUL/engine-predictive-maintenance-dataset`) |
| **EDA** | Univariate, bivariate, multivariate analysis + Mutual Information |
| **Preprocessing** | IQR outlier capping, StandardScaler, 3 engineered features |
| **Class Balancing** | SMOTE on training data |
| **Model Training** | 6 algorithms: DecisionTree, RandomForest, GradientBoosting, XGBoost, AdaBoost, LightGBM |
| **Tuning** | RandomizedSearchCV with 5-fold StratifiedKFold |
| **Tracking** | MLflow (10 metrics per model) |
| **Explainability** | SHAP TreeExplainer |
| **Quality Gate** | F1 > 0.60, AUC > 0.65 |
| **Deployment** | Streamlit on HuggingFace Spaces |
| **CI/CD** | GitHub Actions (auto-trigger on push to main) |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app.git
cd WILDESOUL-engine-maintenance-app

# Install
pip install -r deployment/requirements.txt
pip install mlflow matplotlib seaborn imbalanced-learn shap lightgbm jupytext

# Train models
python predictive_maintenance.py

# Run app locally
cd deployment && streamlit run app.py
```

## 🔄 CI/CD Automation

Every push to `main` automatically:
1. Installs dependencies
2. Converts `.py` → `.ipynb` via jupytext
3. Executes the notebook (trains 6 models)
4. Deploys the Streamlit app to HuggingFace Spaces

## 📊 Live Demo

👉 **[Try the Streamlit App](https://huggingface.co/spaces/WILDESOUL/engine-maintenance-app)**

---

*AIML Capstone Project — Predictive Maintenance © 2026 WildeSoul*
