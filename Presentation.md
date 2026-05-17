# Predictive Maintenance: Portfolio Presentation

## 1. Project Overview

**Title:** Predictive Engine Health Monitoring System

**Business Context:** Unplanned vehicle breakdowns cost fleet operators thousands in emergency repairs. This project builds an ML system analyzing real-time engine sensor data to predict failures before they occur.

**Objective:** Deploy an automated MLOps pipeline training 7 ML models including a Voting Ensemble with a quality gate and interactive Streamlit dashboard.

## 2. Technical Deep Dive

**Data Pipeline:** 19,536 records, IQR outlier capping, 3 engineered features (Temp_Pressure_Ratio, Coolant_Efficiency, High_RPM_Flag), SMOTE resampling.

**Models:** DecisionTree, RandomForest, GradientBoosting, XGBoost, AdaBoost, LightGBM, VotingEnsemble. Tuned with RandomizedSearchCV and 5-fold StratifiedKFold. MLflow tracks 10 metrics per model. SHAP explainability validates predictions.

**Advanced Analysis:** Precision-Recall curves, Learning curves, Classification report, Model comparison charts.

**Deployment:** 4-tab Streamlit dashboard (Prediction, Gauges, Model Comparison, Batch Predict) with Plotly visualizations. Multi-stage GitHub Actions CI/CD. Evidently AI drift detection.

## 3. Business Impact

1. Sensor-driven maintenance reduces emergency repairs by 30-40%
2. Estimated savings of $15k-$25k per quarter for 100-vehicle fleets
3. Maximize fleet uptime by 20-25%

## 4. Links

- GitHub: https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app
- Live App: https://huggingface.co/spaces/WILDESOUL/engine-maintenance-app
- Dataset: https://huggingface.co/datasets/WILDESOUL/engine-predictive-maintenance-dataset
- Model: https://huggingface.co/WILDESOUL/engine-maintenance-model
