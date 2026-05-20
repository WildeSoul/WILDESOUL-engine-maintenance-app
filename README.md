# Predictive Engine Maintenance — PGP Capstone Project

[![GitHub Actions](https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app/actions/workflows/pipeline.yml/badge.svg)](https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app/actions)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Live%20App-yellow)](https://huggingface.co/spaces/WILDESOUL/engine-maintenance-app)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)](https://pytorch.org)
[![LoRA](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)

## Architecture

```
[ 6 Engine Sensors ]
        |
        v
[ Advanced Feature Engineering ]
  - Time-Domain: RMS, Kurtosis, Skewness, Crest Factor
  - Frequency-Domain: FFT Magnitudes, Spectral Energy, Centroid
  - Cross-Sensor: Thermal Load, Pressure Gradient, Z-Scores
  = 31 Total Features
        |
        v
[ FT-Transformer (Feature Tokenizer Transformer) ]
  - Feature Tokenization: Each feature -> 64-dim embedding
  - Transformer Encoder: 3 layers, 4 heads
  - LoRA Fine-Tuning on Q, V matrices (rank=8, alpha=32)
  - 91% parameter reduction (115K total -> 10K trainable)
        |
        v
[ Multi-Task Output ]
  - Classification: Normal / Faulty
  - Severity: Failure risk score [0, 1]
        |
        v
[ SHAP Explainability Dashboard ]
  - Per-prediction sensor contribution
  - Global feature importance
  - Integrated in Streamlit Control Room
```

## Key Features

| Feature | Details |
|---------|---------|
| **FT-Transformer + LoRA** | Parameter-efficient fine-tuning with 91% reduction |
| **31 Engineered Features** | Signal processing (FFT, RMS, Kurtosis) + domain-specific |
| **7 Traditional ML Models** | DecisionTree, RF, GBM, XGBoost, AdaBoost, LightGBM, VotingEnsemble |
| **SHAP Explainability** | Sensor-level contribution analysis |
| **NASA Asymmetric Scoring** | Business-aware metric (penalizes missed failures 3x) |
| **Enterprise Metrics** | Macro F1, PR-AUC, MCC, Cost-Weighted Analysis |
| **7-Tab Control Room** | Prediction, Sensors, SHAP, Comparison, LoRA, Fleet, Batch |
| **CI/CD Pipeline** | GitHub Actions -> auto-deploy to HuggingFace Spaces |
| **MLflow Tracking** | LoRA hyperparameter experiments logged |

## Tech Stack

- **Deep Learning:** PyTorch, Custom FT-Transformer, LoRA
- **Traditional ML:** scikit-learn, XGBoost, LightGBM
- **Feature Engineering:** scipy (FFT), numpy
- **Explainability:** SHAP (KernelExplainer)
- **Experiment Tracking:** MLflow
- **Deployment:** Streamlit on HuggingFace Spaces
- **CI/CD:** GitHub Actions
- **GPU:** NVIDIA RTX 4060

## Quick Start

```bash
# Train traditional models (runs in CI/CD automatically)
python predictive_maintenance.py

# Train transformer with LoRA (requires GPU)
python train_transformer.py

# Launch dashboard locally
streamlit run deployment/app.py
```

## Project Structure

```
WILDESOUL-engine-maintenance-app/
├── .github/workflows/pipeline.yml    # CI/CD pipeline
├── data/engine_data.csv              # Raw dataset (19,535 rows)
├── deployment/
│   ├── app.py                        # 7-tab Control Room Dashboard
│   ├── requirements.txt              # Dependencies
│   ├── README.md                     # HF Space config
│   └── Dockerfile                    # Container config
├── model_building/
│   ├── best_model.joblib             # Traditional ML pipeline
│   ├── transformer_model.pt          # FT-Transformer + LoRA weights
│   ├── transformer_scaler.joblib     # Feature scaler
│   ├── evaluation_report.json        # Enterprise metrics
│   ├── training_history.json         # Loss/F1 curves
│   ├── param_stats.json              # LoRA parameter stats
│   └── plots/                        # Visualizations
├── config.py                         # Centralized configuration
├── advanced_features.py              # Signal processing features
├── transformer_model.py              # FT-Transformer + LoRA architecture
├── train_transformer.py              # GPU training script
├── evaluation_metrics.py             # Enterprise metrics (NASA, PR-AUC)
├── predictive_maintenance.py         # Traditional ML pipeline
├── VIVA_NOTES.md                     # Viva preparation
├── Final_Report.md                   # Business report
└── README.md                         # This file
```

## License

Apache 2.0

---

> **PGP in AIML Capstone** | 2026 WildeSoul
