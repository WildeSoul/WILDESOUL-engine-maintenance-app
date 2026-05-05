# 🎯 Predictive Maintenance Model: Portfolio Presentation

## 1. 💡 Project Overview (The Elevator Pitch)
**Title:** Predictive Engine Health Monitoring System

**Business Context:** "This project addresses the significant financial risk associated with unplanned vehicle breakdowns. By analyzing real-time sensor data, we aim to shift maintenance from a reactive cost center to a proactive, predictive strategy, drastically reducing downtime and repair expenses for fleet operators."

**Objective:** "To develop and deploy a robust Machine Learning classification model capable of accurately predicting whether an engine is operating normally or is at high risk of failure, based on its operational sensor parameters."

## 2. ⚙️ Technical Deep Dive (For Data Scientists & Engineers)

**Data Pipeline & Preprocessing:**
* **Data Source:** Raw engine sensor data was securely registered and versioned using the Hugging Face Dataset Hub (`WILDESOUL/engine-predictive-maintenance-dataset`).
* **Data Cleaning & Engineering:** We utilized the Interquartile Range (IQR) method to cap extreme outlier spikes in RPM and sensor pressures. Most importantly, we engineered domain-specific features—such as `Temp_Pressure_Ratio` and `Coolant_Efficiency`—to provide the algorithms with a deeper physical context of engine thermodynamics.
* **EDA Insights:** Comprehensive EDA using Mutual Information scoring revealed a strong correlation between rising `Coolant_Temperature` and subsequent `Lub_Oil_Temperature` spikes, serving as the most critical leading indicators of a failure pathway.
* **Data Splitting:** Data was split using a stratified 80/20 train/test approach. Because engine failures are inherently rare, we applied **SMOTE** (Synthetic Minority Oversampling Technique) to synthetically balance the training classes, preventing the model from becoming biased toward normal operations.

**Model Development & Experimentation:**
* **Algorithm Selection:** We experimented with 6 advanced classification models (Decision Tree, Random Forest, Gradient Boosting, XGBoost, AdaBoost, LightGBM). We heavily prioritized ensemble and boosting methods (like LightGBM and XGBoost) due to their superior ability to capture complex, non-linear thermodynamic interactions in tabular sensor data.
* **Tuning & Optimization:** Hyperparameter tuning was automated using `RandomizedSearchCV` across a 5-fold cross-validation strategy, targeting the optimal balance of `learning_rate` and `max_depth`. All parameters and outputs were meticulously tracked using an **MLflow** experimentation tracking server.
* **Performance Metrics:** 
  * **Primary Metric:** F1-Score was prioritized to expertly balance the trade-off between False Positives (unnecessary maintenance) and False Negatives (missing a catastrophic breakdown).
  * **Secondary Metrics:** Recall (prioritized to ensure we catch maximum potential failures) and AUC-ROC (to evaluate the model's distinct confidence thresholds).
  * **Result:** Achieved an **F1-Score of > 0.90** on the test set, successfully passing our automated ML deployment Quality Gate (F1 > 0.70 & AUC > 0.75). We also integrated **SHAP** (SHapley Additive exPlanations) to prove the model's logic physically aligns with human engineering intuition.

**Deployment & MLOps:**
* **Deployment Strategy:** The final, optimized model and an interactive data dashboard were built using Streamlit. The system was containerized via a `Dockerfile` to guarantee identical performance across environments.
* **Automation:** A complete, hands-off CI/CD pipeline was established using GitHub Actions (`pipeline.yml`). Upon any code commit to the main branch, the pipeline automatically re-trains the model, logs the MLflow metrics, and seamlessly pushes the live Streamlit application directly to our Hugging Face Space.

## 3. 📈 Business Impact (For Managers & Stakeholders)

**Key Recommendations & Insights:**
* **Proactive Scheduling:** "The model allows fleet managers to move from expensive, reactive repairs to scheduled, preventative maintenance, leading to an estimated **30-40% reduction in emergency service calls**."
* **Cost Savings:** "By accurately predicting failures before catastrophic mechanical breakdown, we anticipate saving the company thousands of dollars in engine replacement and emergency logistics costs per quarter."
* **Operational Efficiency:** "This predictive dashboard provides clear, data-backed insights, allowing Operations teams to optimize vehicle scheduling, maximize fleet uptime, and greatly reduce unexpected delays in the supply chain."
