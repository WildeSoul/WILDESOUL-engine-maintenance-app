# %% [markdown]
# # Predictive Maintenance - Engine Health Monitoring
# 
# ## Business Context
# Vehicle breakdowns and engine failures lead to significant financial losses. Predictive maintenance helps minimize these issues by leveraging sensor data to forecast potential failures before they occur.

# %% [markdown]
# ## 1. Setup and Master Folders Creation

# %%
import os
import warnings
warnings.filterwarnings('ignore')

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
import numpy as np
from huggingface_hub import HfApi
try:
    from google.colab import userdata
except ImportError:
    pass

try:
    HF_TOKEN = userdata.get('HF_TOKEN')
except:
    HF_TOKEN = os.environ.get('HF_TOKEN')

# Read actual dataset
if os.path.exists('data/engine_data.csv'):
    df = pd.read_csv('data/engine_data.csv')
    print("Loaded actual dataset 'data/engine_data.csv'.")
else:
    print("Dataset 'data/engine_data.csv' not found. Please provide the dataset.")

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

# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

sns.set_theme(style='whitegrid', palette='muted')

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
num_cols = df.select_dtypes(include=[np.number]).columns.drop('Engine_Condition')
fig, axes = plt.subplots((len(num_cols)+1)//2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.histplot(data=df, x=col, hue='Engine_Condition', kde=True, ax=axes[i], palette={0:'#2ecc71', 1:'#e74c3c'})
    axes[i].set_title(f'Distribution of {col}')
fig.tight_layout()
plt.savefig('model_building/plots/feature_distributions.png', bbox_inches='tight')
plt.close()

# 4. Outlier Boxplots
fig, axes = plt.subplots((len(num_cols)+1)//2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, y=col, x='Engine_Condition', ax=axes[i], palette={0:'#2ecc71', 1:'#e74c3c'})
    axes[i].set_title(f'Boxplot of {col}')
fig.tight_layout()
plt.savefig('model_building/plots/outlier_boxplots.png', bbox_inches='tight')
plt.close()

# 5. Feature Importance (Mutual Information)
X_mi = df.drop('Engine_Condition', axis=1)
mi_scores = mutual_info_classif(X_mi, df['Engine_Condition'], random_state=42)
mi_df = pd.DataFrame({'Feature': X_mi.columns, 'MI_Score': mi_scores}).sort_values('MI_Score', ascending=True)
plt.figure(figsize=(8, 6))
plt.barh(mi_df['Feature'], mi_df['MI_Score'], color=plt.cm.viridis(np.linspace(0.2, 0.9, len(mi_df))))
plt.xlabel('Mutual Information Score')
plt.title('Feature Importance (Mutual Information)')
plt.savefig('model_building/plots/feature_importance_mi.png', bbox_inches='tight')
plt.close()

print("EDA completed and plots saved.")

# %% [markdown]
# ## 4. Feature Engineering & Data Preparation

# %%
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Feature Engineering: Derived features
df['Temp_Pressure_Ratio'] = df['Lub_Oil_Temperature'] / df['Lub_Oil_Pressure'].replace(0, np.nan)
df['Temp_Pressure_Ratio'] = df['Temp_Pressure_Ratio'].fillna(df['Temp_Pressure_Ratio'].median())
df['Coolant_Efficiency'] = df['Coolant_Pressure'] / df['Coolant_Temperature'].replace(0, np.nan)
df['Coolant_Efficiency'] = df['Coolant_Efficiency'].fillna(df['Coolant_Efficiency'].median())
df['High_RPM_Flag'] = (df['Engine_RPM'] > df['Engine_RPM'].quantile(0.85)).astype(int)

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

num_features_to_cap = ['Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature']
df_cleaned = cap_outliers_iqr(df, num_features_to_cap)

X = df_cleaned.drop('Engine_Condition', axis=1)
y = df_cleaned['Engine_Condition']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# SMOTE for Class Imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f'Before SMOTE: {dict(y_train.value_counts())}')
print(f'After SMOTE:  {dict(y_train_resampled.value_counts())}')

train_df = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

if HF_TOKEN:
    api.upload_file(path_or_fileobj='data/train.csv', path_in_repo='train.csv', repo_id=dataset_repo, repo_type='dataset', token=HF_TOKEN)
    api.upload_file(path_or_fileobj='data/test.csv', path_in_repo='test.csv', repo_id=dataset_repo, repo_type='dataset', token=HF_TOKEN)

# %% [markdown]
# ## 5. Model Building, MLflow Tracking, and SHAP Explainability

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import mlflow, mlflow.sklearn
import joblib, json, shap

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_config = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth':[5,10,15,None]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators':[100,200,300], 'max_depth':[5,10,15,None]}
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators':[100,200], 'learning_rate':[0.05, 0.1]}
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss'),
        'params': {'n_estimators':[100,200], 'learning_rate':[0.05, 0.1], 'max_depth':[3,5,7]}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42, algorithm='SAMME'),
        'params': {'n_estimators':[50,100,200], 'learning_rate':[0.05, 0.1, 0.5]}
    },
    'LightGBM': {
        'model': LGBMClassifier(random_state=42, verbose=-1),
        'params': {'n_estimators':[100,200], 'max_depth':[5,10,-1], 'learning_rate':[0.05, 0.1]}
    }
}

mlflow.set_experiment('Engine_Predictive_Maintenance')

best_model = None; best_score = 0; best_model_name = ''
results = {}; results_proba = {}; all_models = {}

for name, config in models_config.items():
    with mlflow.start_run(run_name=name):
        search = RandomizedSearchCV(config['model'], config['params'], cv=cv_strategy, scoring='f1', n_iter=5, random_state=42)
        search.fit(X_train_resampled, y_train_resampled)
        tuned = search.best_estimator_
        
        # Evaluate on Train
        y_train_pred = tuned.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred, zero_division=0)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)

        # Evaluate on Test
        y_test_pred = tuned.predict(X_test_scaled)
        y_test_proba = tuned.predict_proba(X_test_scaled)[:, 1] if hasattr(tuned, "predict_proba") else y_test_pred
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        mlflow.log_params(search.best_params_)
        mlflow.log_param('smote_applied', True)
        mlflow.log_metrics({
            'train_accuracy': train_accuracy, 'train_precision': train_precision,
            'train_recall': train_recall, 'train_f1-score': train_f1,
            'test_accuracy': test_accuracy, 'test_precision': test_precision,
            'test_recall': test_recall, 'test_f1-score': test_f1,
            'auc_roc': test_auc, 'cv_best_score': search.best_score_
        })
        
        mlflow.sklearn.log_model(tuned, name)
        
        results[name] = {'f1_score': test_f1, 'auc_roc': test_auc, 'cv_best_score': search.best_score_}
        results_proba[name] = y_test_proba
        all_models[name] = tuned
        print(f"{name} -> Test F1 Score: {test_f1:.4f} | AUC: {test_auc:.4f}")
        
        if test_f1 > best_score:
            best_score = test_f1; best_model = tuned; best_model_name = name

print(f"\nBest Model: {best_model_name} (F1: {best_score:.4f})")

# %% [markdown]
# ## 6. Model Evaluation Plots & JSON Export

# %%
# Confusion Matrix for Best Model
best_y_pred = all_models[best_model_name].predict(X_test_scaled)
cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Faulty'], yticklabels=['Normal', 'Faulty'])
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Confusion Matrix — {best_model_name}')
plt.savefig('model_building/plots/confusion_matrix.png', bbox_inches='tight')
plt.close()

# ROC Curves for all Models
plt.figure(figsize=(10, 8))
colors = plt.cm.Set1(np.linspace(0, 1, len(results_proba)))
for (name, y_p), color in zip(results_proba.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, y_p)
    plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={roc_auc_score(y_test, y_p):.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.savefig('model_building/plots/roc_curves.png', bbox_inches='tight')
plt.close()

# SHAP Explainability
try:
    explainer = shap.TreeExplainer(all_models[best_model_name])
    shap_values = explainer.shap_values(X_test_scaled)
    if isinstance(shap_values, list): shap_values_plot = shap_values[1]
    else: shap_values_plot = shap_values
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_plot, X_test_scaled, show=False)
    plt.title('SHAP Summary Plot')
    plt.savefig('model_building/plots/shap_summary.png', bbox_inches='tight')
    plt.close()
except Exception as e:
    print(f"SHAP Error: {e}")

# JSON Export
from sklearn.pipeline import Pipeline
# Bundle scaler and best model for deployment
deploy_pipeline = Pipeline([('scaler', scaler), ('model', best_model)])
joblib.dump(deploy_pipeline, 'model_building/best_model.joblib')

feature_info = {
    'feature_names': list(X_train.columns),
    'best_model_name': best_model_name,
    'best_f1_score': best_score,
    'smote_applied': True,
    'engineered_features': ['Temp_Pressure_Ratio', 'Coolant_Efficiency', 'High_RPM_Flag']
}
with open('model_building/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

with open('model_building/model_comparison.json', 'w') as f:
    json.dump({'best_model': best_model_name, 'results': results}, f, indent=2)

# Model Quality Gate
if best_score < 0.70 or results[best_model_name]['auc_roc'] < 0.75:
    raise ValueError(f"Quality Gate Failed: Model F1 ({best_score:.4f}) or AUC too low.")
print("Model passed Quality Gate!")

if HF_TOKEN:
    api.upload_file(path_or_fileobj='model_building/best_model.joblib', path_in_repo='best_model.joblib', repo_id=model_repo, token=HF_TOKEN)
    api.upload_file(path_or_fileobj='model_building/feature_info.json', path_in_repo='feature_info.json', repo_id=model_repo, token=HF_TOKEN)
    api.upload_file(path_or_fileobj='model_building/model_comparison.json', path_in_repo='model_comparison.json', repo_id=model_repo, token=HF_TOKEN)
    import glob
    for fpath in glob.glob('model_building/plots/*.png'):
        api.upload_file(path_or_fileobj=fpath, path_in_repo=f'plots/{os.path.basename(fpath)}', repo_id=model_repo, token=HF_TOKEN)
