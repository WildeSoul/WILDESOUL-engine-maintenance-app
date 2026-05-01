# Interim Business Report: Predictive Maintenance for Engine Health

## 1. Data Registration
A master folder named `predictive_maintenance` was created, housing a subfolder `data`. We successfully registered and loaded the dataset containing key engine health parameters (`Engine_RPM`, `Lub_Oil_Pressure`, `Fuel_Pressure`, `Coolant_Pressure`, `Lub_Oil_Temperature`, `Coolant_Temperature`, and `Engine_Condition`) onto the Hugging Face dataset space.

## 2. Exploratory Data Analysis (EDA)
**Data Overview:** The dataset models common operating parameters for both small and large engines. The target variable is `Engine_Condition` (0: Normal, 1: Faulty).

* **Univariate Analysis:** Most numerical parameters exhibited a relatively normal distribution, with right-skewed tails indicating high-stress engine conditions.
* **Bivariate & Multivariate Analysis:** 
   - A correlation heatmap revealed strong relationships between elevated temperatures (both coolant and oil) and the likelihood of engine failure. 
   - Mutual Information (MI) scores indicated that `Engine_RPM` and `Lub_Oil_Temperature` are the most critical predictors of engine health.
* **Insights:** The EDA confirmed that unexpected temperature spikes and abnormal RPM fluctuations act as leading indicators for engine failure.

## 3. Data Preparation
- **Loading:** Data was pulled directly from the Hugging Face space.
- **Cleaning & Feature Engineering:** We applied the Interquartile Range (IQR) method to cap extreme outliers in RPM and pressure metrics. We engineered derived features such as `Temp_Pressure_Ratio` and `Coolant_Efficiency` to give the models deeper physical context.
- **Splitting & Resampling:** The dataset was split using an 80/20 train-test split. To address class imbalance (since failure events are rarer than normal operations), we applied **SMOTE** to synthetically balance the training set. The finalized splits were saved locally and re-uploaded to Hugging Face.

## 4. Model Building with Experimentation Tracking
We employed 6 distinct machine learning algorithms: `DecisionTree`, `RandomForest`, `GradientBoosting`, `XGBoost`, `AdaBoost`, and `LightGBM`.
- **Hyperparameter Tuning:** We utilized `RandomizedSearchCV` with 5-fold cross-validation to find the optimal parameters for each model.
- **MLflow Tracking:** All parameters, along with 8 distinct classification metrics (Accuracy, Precision, Recall, and F1-Score for both Train and Test sets), and the AUC-ROC were systematically logged in MLflow.
- **Best Model:** `GradientBoosting` emerged as the optimal model, successfully balancing precision and recall to predict engine faults without excessive false alarms. The best model was registered into the Hugging Face model hub.
