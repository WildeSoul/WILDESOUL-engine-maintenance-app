# %% [markdown]
# # Predictive Maintenance - Engine Health Monitoring
# 
# ## Business Context
# Vehicle breakdowns and engine failures lead to significant financial losses. Unexpected engine failures can cause expensive repairs, operational downtime, and safety risks. Predictive maintenance helps minimize these issues by leveraging sensor data to forecast potential failures before they occur.

# %% [markdown]
# ## 1. Setup and Master Folders Creation

# %%
import os
os.makedirs('data', exist_ok=True)
os.makedirs('model_building', exist_ok=True)
os.makedirs('model_building/plots', exist_ok=True)
os.makedirs('deployment', exist_ok=True)
os.makedirs('.github/workflows', exist_ok=True)
print("Master folder and subfolders created successfully.")

# %% [markdown]
# ## 2. Data Registration on Hugging Face Space

# %%
import pandas as pd
from huggingface_hub import HfApi
try:
    from google.colab import userdata
except ImportError:
    pass

# Replace with your actual HF Token if running locally
# os.environ['HF_TOKEN'] = 'YOUR_HF_TOKEN'
try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except:
    HF_TOKEN = os.environ.get('HF_TOKEN')

# Create mock data if it doesn't exist
if not os.path.exists('data/engine_data.csv'):
    import numpy as np
    np.random.seed(42)
    n = 2000
    rpm = np.random.normal(2500, 500, n)
    lub_oil_pressure = np.random.normal(3.5, 0.5, n)
    fuel_pressure = np.random.normal(4.0, 0.4, n)
    coolant_pressure = np.random.normal(2.0, 0.3, n)
    lub_oil_temp = np.random.normal(85, 10, n)
    coolant_temp = np.random.normal(80, 8, n)
    
    # Introduce correlation with target (1 = Faulty, 0 = Normal)
    condition_prob = 1 / (1 + np.exp(-(
        (rpm - 2500)/500 * 0.5 
        - (lub_oil_pressure - 3.5)/0.5 * 1.5 
        - (fuel_pressure - 4.0)/0.4 * 1.0 
        + (lub_oil_temp - 85)/10 * 1.2 
        + (coolant_temp - 80)/8 * 1.2
    )))
    condition = np.where(condition_prob > 0.5, 1, 0)
    
    df = pd.DataFrame({
        'Engine_RPM': rpm, 'Lub_Oil_Pressure': lub_oil_pressure,
        'Fuel_Pressure': fuel_pressure, 'Coolant_Pressure': coolant_pressure,
        'Lub_Oil_Temperature': lub_oil_temp, 'Coolant_Temperature': coolant_temp,
        'Engine_Condition': condition
    })
    
    # Add outliers
    outlier_indices = np.random.choice(n, size=50, replace=False)
    df.loc[outlier_indices, 'Engine_RPM'] = np.random.uniform(5000, 6000, 50)
    df.loc[outlier_indices, 'Lub_Oil_Temperature'] = np.random.uniform(120, 150, 50)
    df.loc[outlier_indices, 'Engine_Condition'] = 1
    
    df.to_csv('data/engine_data.csv', index=False)
    print("Generated mock dataset 'data/engine_data.csv'.")

# If you have an HF token, you can push the dataset to HF:
if HF_TOKEN:
    api = HfApi()
    user_info = api.whoami(token=HF_TOKEN)
    HF_USERNAME = user_info['name']
    dataset_repo = f'{HF_USERNAME}/engine-predictive-maintenance-dataset'
    api.create_repo(repo_id=dataset_repo, repo_type='dataset', exist_ok=True, token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj='data/engine_data.csv',
        path_in_repo='engine_data.csv',
        repo_id=dataset_repo,
        repo_type='dataset',
        token=HF_TOKEN
    )
    print(f"Dataset registered at: https://huggingface.co/datasets/{dataset_repo}")
else:
    print("HF_TOKEN not set. Skipping Hugging Face dataset registration.")

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
df = pd.read_csv('data/engine_data.csv')

# 1. Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Engine_Condition', hue='Engine_Condition', palette={0:'#2ecc71', 1:'#e74c3c'}, legend=False)
plt.title('Target Variable Distribution (Engine Condition)')
plt.xticks([0, 1], ['Normal (0)', 'Faulty (1)'])
plt.savefig('model_building/plots/target_distribution.png', bbox_inches='tight')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('model_building/plots/correlation_heatmap.png', bbox_inches='tight')
plt.close()

# 3. Distributions of Numerical Features
num_cols = df.columns.drop('Engine_Condition')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Engine_Condition', kde=True, ax=axes[i], palette={0:'#2ecc71', 1:'#e74c3c'})
    axes[i].set_title(col)
fig.tight_layout()
plt.savefig('model_building/plots/feature_distributions.png', bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 4. Data Preparation

# %%
from sklearn.model_selection import train_test_split

# Outlier handling (IQR)
def cap_outliers_iqr(data, columns, factor=1.5):
    data_cleaned = data.copy()
    for col in columns:
        Q1 = data_cleaned[col].quantile(0.25)
        Q3 = data_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        data_cleaned[col] = data_cleaned[col].clip(lower=lower, upper=upper)
    return data_cleaned

df_cleaned = cap_outliers_iqr(df, num_cols)

X = df_cleaned.drop('Engine_Condition', axis=1)
y = df_cleaned['Engine_Condition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)
print("Data Preparation complete. Saved train.csv and test.csv")

if HF_TOKEN:
    api.upload_file(path_or_fileobj='data/train.csv', path_in_repo='train.csv', repo_id=dataset_repo, repo_type='dataset', token=HF_TOKEN)
    api.upload_file(path_or_fileobj='data/test.csv', path_in_repo='test.csv', repo_id=dataset_repo, repo_type='dataset', token=HF_TOKEN)

# %% [markdown]
# ## 5. Model Building with Experimentation Tracking

# %%
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import mlflow, mlflow.sklearn
import joblib, json

models_config = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators':[100,200], 'max_depth':[5,10,None]}
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators':[100,200], 'learning_rate':[0.05, 0.1]}
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
        'params': {'n_estimators':[100,200], 'learning_rate':[0.05, 0.1], 'max_depth':[3,5,7]}
    }
}

# Start MLflow locally if possible (skip if on colab without setup, but here we track in memory/files)
mlflow.set_experiment('Engine_Predictive_Maintenance')

best_model = None
best_score = 0
best_model_name = ''
results = {}

for name, config in models_config.items():
    with mlflow.start_run(run_name=name):
        search = RandomizedSearchCV(config['model'], config['params'], cv=3, scoring='f1', n_iter=3, random_state=42)
        search.fit(X_train, y_train)
        tuned = search.best_estimator_
        
        y_pred = tuned.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_params(search.best_params_)
        mlflow.log_metric('f1_score', f1)
        mlflow.sklearn.log_model(tuned, name)
        
        results[name] = f1
        print(f"{name} F1 Score: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = tuned
            best_model_name = name

print(f"\nBest Model: {best_model_name} (F1: {best_score:.4f})")

model_path = 'model_building/best_model.joblib'
joblib.dump(best_model, model_path)

if HF_TOKEN:
    model_repo = f'{HF_USERNAME}/engine-maintenance-model'
    api.create_repo(repo_id=model_repo, exist_ok=True, token=HF_TOKEN)
    api.upload_file(path_or_fileobj=model_path, path_in_repo='best_model.joblib', repo_id=model_repo, token=HF_TOKEN)
    print(f"Model registered at: https://huggingface.co/{model_repo}")
