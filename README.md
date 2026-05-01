# Predictive Maintenance - Engine Health Monitoring 🚀

## Overview
This project implements an end-to-end Machine Learning pipeline for Predictive Maintenance of engines, identifying potential failures based on sensor data (RPM, Pressure, Temperature, etc.).

## Project Structure
- `data/`: Contains the datasets (`engine_data.csv`, `train.csv`, `test.csv`).
- `model_building/`: Contains the final trained model (`best_model.joblib`), EDA plots, and MLflow tracking logs.
- `deployment/`: Contains the Streamlit web application `app.py`, `Dockerfile`, and `requirements.txt` for deploying on Hugging Face Spaces.
- `.github/workflows/`: Contains the `pipeline.yml` for automated CI/CD and deployment to Hugging Face via GitHub Actions.
- `predictive_maintenance.ipynb`: The primary Jupyter Notebook covering Data Registration, EDA, Preprocessing, and Model Training with MLflow experimentation tracking.

## Pipeline Features
1. **Data Registration**: Mock data generation based on Engine parameters, ready to be registered on Hugging Face Datasets.
2. **Exploratory Data Analysis (EDA)**: Visualizations of feature distributions, correlation heatmaps, and target distribution.
3. **Data Preparation**: Handling missing values, outlier capping using the IQR method, and Train/Test splits.
4. **Model Building & MLflow Tracking**: Hyperparameter tuning for RandomForest, GradientBoosting, and XGBoost using `RandomizedSearchCV`, tracked automatically using MLflow.
5. **Deployment**: Containerized with Docker and served via Streamlit.

## Quick Start
1. Ensure you have installed requirements: `pip install -r deployment/requirements.txt`
2. Run the notebook `predictive_maintenance.ipynb` to generate the `best_model.joblib` and datasets.
3. Run the Streamlit app locally: `cd deployment && streamlit run app.py`

## Automation
The `.github/workflows/pipeline.yml` file is configured to run tests and push the deployment artifacts to Hugging Face Space automatically upon push to the `main` branch.
