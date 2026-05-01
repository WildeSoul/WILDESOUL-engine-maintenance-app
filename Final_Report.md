# Final Business Report: Predictive Maintenance for Engine Health

## 1. Introduction & Data Registration
Vehicle breakdowns lead to massive operational and financial losses. By utilizing historical engine sensor data, we constructed an automated pipeline to predict engine failures. The raw data was registered into the Hugging Face dataset space within a structured `data` folder architecture.

## 2. Exploratory Data Analysis (EDA)
Comprehensive EDA revealed that `Engine_RPM` and internal temperatures are the strongest predictors of engine failure. Visualizations (Mutual Information plots, boxplots) highlighted that engines operating beyond nominal temperature thresholds are exponentially more likely to be classified as 'Faulty'.

## 3. Data Preparation
Data was pulled from Hugging Face and cleaned. Outliers were capped using the IQR technique. To maximize model accuracy, we generated domain-specific features: `Temp_Pressure_Ratio` and `Coolant_Efficiency`. Due to the rarity of engine faults, we balanced the training data using **SMOTE**. The processed splits were versioned on Hugging Face.

## 4. Model Building & Tracking
We executed a comprehensive hyperparameter tuning process across 6 advanced algorithms (including XGBoost, LightGBM, and Random Forest). **MLflow** was used to rigorously log all parameters and track 8 evaluation metrics (Train/Test Accuracy, Precision, Recall, F1). The top-performing algorithm was registered in the Hugging Face Model Hub, guided by a strict quality gate ensuring an F1 score > 0.70. Additionally, **SHAP** explainability was used to verify that the model's decision-making aligned with real-world physical constraints.

## 5. Model Deployment
We developed a production-ready **Streamlit web application**. A custom `Dockerfile` and `requirements.txt` were defined to containerize the app. The application seamlessly loads the trained model from Hugging Face, accepts real-time sensor inputs, dynamically calculates the derived features, and outputs an immediate prediction on engine health.

## 6. Automated GitHub Actions Workflow
The entire MLOps pipeline is fully automated via `.github/workflows/pipeline.yml`. Upon any code push to the `main` branch, GitHub Actions provisions an environment, installs all dependencies, executes the Jupyter notebook to retrain the model on fresh data, and autonomously pushes the deployment files directly to the Hugging Face Space.

## 7. Output Evaluation
- **GitHub Repository:** Contains the complete structured workflow, code, and automated pipeline.
- **Hugging Face Space:** Hosts the live, interactive Streamlit application.
*(Reviewers: Please refer to the repository links and attached screenshots for execution validation).*

## 8. Actionable Insights and Recommendations
1. **Real-time Monitoring:** Fleet managers should integrate this Streamlit API directly into vehicle dashboards. Since the model prioritizes RPM and Temperature anomalies, real-time alerts can stop engines before catastrophic failure occurs.
2. **Maintenance Scheduling:** Instead of relying on static mileage-based schedules, operators can transition to a purely predictive maintenance schedule, saving costs on premature servicing while preventing unexpected downtime.
3. **Sensor Calibration:** As `Temp_Pressure_Ratio` is highly predictive, hardware teams should ensure temperature and pressure sensors are recalibrated frequently to guarantee high-quality data feeds into the model.
