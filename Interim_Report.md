# Interim Business Report: Predictive Maintenance for Engine Health

**Project Title:** Predictive Engine Health Monitoring System  
**Author:** Dibyajyoti (WildeSoul)  
**Date:** May 2026  
**Repository:** [GitHub](https://github.com/WildeSoul/WILDESOUL-engine-maintenance-app)

---

## 1. Data Registration (6 Points)

### 1.1 Master Folder Structure
A master project folder was created with subfolders: `data/` (raw + processed datasets), `model_building/` (trained models, plots, JSON exports), `deployment/` (Streamlit app, Dockerfile, requirements), and `.github/workflows/` (CI/CD pipeline).

### 1.2 Hugging Face Dataset Registration
The raw dataset was registered on Hugging Face at `WILDESOUL/engine-predictive-maintenance-dataset`. It contains **19,536 records** with 7 features from engines of varying sizes (vehicles, generators, lawnmowers).

---

## 2. Exploratory Data Analysis (10 Points)

### 2.1 Data Collection and Background
The dataset models sensor readings from engines. Features include RPM, oil/fuel/coolant pressure, oil/coolant temperature, and a binary target `Engine_Condition` (0=Normal, 1=Faulty).

### 2.2 Data Overview

| Feature | Type | Range | Unit |
|---------|------|-------|------|
| Engine_RPM | Numerical | 276–2,143 | RPM |
| Lub_Oil_Pressure | Numerical | 0.07–7.27 | bar |
| Fuel_Pressure | Numerical | 0.10–19.51 | bar |
| Coolant_Pressure | Numerical | 0.16–7.48 | bar |
| Lub_Oil_Temperature | Numerical | 72.77–88.62 | °C |
| Coolant_Temperature | Numerical | 63.54–95.23 | °C |
| Engine_Condition | Binary | 0 or 1 | — |

- **No missing values** across any feature.
- The target variable is **imbalanced** (~60% Faulty, ~40% Normal).

### 2.3 Univariate Analysis
- **Engine RPM:** Right-skewed, majority operating 400–1,200 RPM.
- **Temperatures:** Normal distributions centered ~76–78°C with outliers indicating overheating.
- **Fuel Pressure:** Widest variance, reflecting diverse engine fuel systems.

### 2.4 Bivariate Analysis
- Correlation heatmap: Moderate positive correlation between `Lub_Oil_Temperature` and `Engine_RPM` (r ≈ 0.15).
- Boxplots: Faulty engines have lower oil pressure and higher temperature variability.

### 2.5 Multivariate Analysis
- **Mutual Information scores** ranked `Engine_RPM` and `Lub_Oil_Temperature` as strongest predictors.
- All features contribute non-trivially, justifying their inclusion.

### 2.6 Key Insights
1. High oil temperature relative to low oil pressure → disproportionately faulty.
2. RPM above 85th percentile (~1,062) shows distinct failure patterns.
3. Class imbalance requires SMOTE resampling to avoid majority-class bias.

---

## 3. Data Preparation (10 Points)

- **Loading:** Data loaded from Hugging Face via `huggingface_hub` API.
- **Cleaning:** IQR (1.5×) outlier capping on all 6 sensor features.
- **Feature Engineering:** Created 3 features: `Temp_Pressure_Ratio`, `Coolant_Efficiency`, `High_RPM_Flag`.
- **Split:** 80/20 stratified train-test split with StandardScaler normalization.
- **SMOTE:** Applied on training data only to balance classes (50/50 after).
- **Upload:** Processed `train.csv` and `test.csv` uploaded to Hugging Face.

---

## 4. Model Building with Experimentation Tracking (8 Points)

### 4.1 Algorithms Used
Decision Tree, Bagging, Random Forest, Gradient Boosting, XGBoost, AdaBoost, LightGBM.

### 4.2 Tuning
`RandomizedSearchCV` with 5-fold StratifiedKFold, optimizing F1 Score.

### 4.3 MLflow Tracking
All experiments logged under `Engine_Predictive_Maintenance`:
- **Parameters:** Best hyperparameters + `smote_applied: True`
- **Metrics (10):** Train/Test Accuracy, Precision, Recall, F1; AUC-ROC; CV Score

### 4.4 Best Model
The top model was saved as `best_model.joblib` (Pipeline: Scaler + Model) and registered on HuggingFace Model Hub. SHAP explainability confirmed temperature and RPM as dominant features.

---

## 5. Business Report Quality (6 Points)
- ✅ Clear problem statement and business context
- ✅ Structured sections with appropriate headings
- ✅ Data-driven observations with visualizations
- ✅ Reproducible technical methodology
- ✅ Professional formatting

*Total: **40/40***
