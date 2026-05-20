# ============================================================================
# train_transformer.py — Training Script for FT-Transformer + LoRA
# ============================================================================
"""
End-to-end training pipeline for the FT-Transformer with LoRA fine-tuning.

This script:
1. Loads and preprocesses the engine sensor dataset
2. Engineers advanced features (time-domain, spectral, interactions)
3. Trains a base FT-Transformer model
4. Applies LoRA adapters and fine-tunes
5. Evaluates with enterprise metrics (NASA scoring, PR-AUC, etc.)
6. Generates SHAP explanations
7. Logs everything to MLflow
8. Saves model weights and artifacts

Usage:
    python train_transformer.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.pytorch

warnings.filterwarnings('ignore')

# Project imports
from config import (
    RAW_DATA_PATH, MODEL_DIR, PLOTS_DIR, RAW_COLUMN_MAP, RAW_SENSOR_COLUMNS,
    TARGET_COLUMN, IQR_FACTOR, TEST_SIZE, RANDOM_STATE, TRANSFORMER_CONFIG,
    LORA_CONFIG, EXPERIMENT_CONFIG, TRANSFORMER_WEIGHTS_PATH, LORA_ADAPTER_PATH,
    ensure_dirs
)
from advanced_features import engineer_all_features
from transformer_model import (
    FTTransformer, apply_lora, get_parameter_stats,
    SensorDataset, MultiTaskLoss
)
from evaluation_metrics import (
    compute_classification_metrics, nasa_asymmetric_score,
    compute_cost_weighted_metrics, generate_full_report, compare_models
)


def load_and_preprocess_data():
    """Load raw data, rename columns, and apply IQR capping."""
    print("📂 Loading dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    df.rename(columns=RAW_COLUMN_MAP, inplace=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    print(f"   Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"   Class distribution: {dict(df[TARGET_COLUMN].value_counts())}")

    # IQR outlier capping
    for col in RAW_SENSOR_COLUMNS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - IQR_FACTOR * IQR, Q3 + IQR_FACTOR * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def prepare_features(df: pd.DataFrame):
    """Engineer features and split into train/test."""
    print("\n🔧 Engineering advanced features...")
    X_features = engineer_all_features(df, RAW_SENSOR_COLUMNS)
    y = df[TARGET_COLUMN].values

    feature_names = X_features.columns.tolist()
    n_features = len(feature_names)
    print(f"   Total features: {n_features}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features.values, y,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE on training data
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"   After SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")

    return (X_train_resampled, y_train_resampled,
            X_test_scaled, y_test,
            scaler, feature_names, n_features)


def train_base_model(model, train_loader, val_loader, config, device):
    """
    Train the base FT-Transformer (all parameters trainable).
    This serves as the 'pretrained' model for LoRA fine-tuning.
    """
    print("\n" + "="*60)
    print("  Phase 1: Training Base FT-Transformer (Full)")
    print("="*60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    loss_fn = MultiTaskLoss(
        cls_weight=config.classification_weight,
        sev_weight=config.severity_weight
    )

    best_val_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'lr': []}

    for epoch in range(config.epochs):
        # ── Training ──
        model.train()
        train_losses = []
        for X_batch, y_batch, sev_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            sev_batch = sev_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            losses = loss_fn(outputs, y_batch, sev_batch)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(losses['total'].item())

        scheduler.step()

        # ── Validation ──
        model.eval()
        val_losses = []
        all_preds, all_labels, all_proba = [], [], []

        with torch.no_grad():
            for X_batch, y_batch, sev_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                sev_batch = sev_batch.to(device)

                outputs = model(X_batch)
                losses = loss_fn(outputs, y_batch, sev_batch)
                val_losses.append(losses['total'].item())

                proba = torch.sigmoid(outputs['logits'])
                preds = (proba >= 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_proba.extend(proba.cpu().numpy())

        val_f1 = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        from sklearn.metrics import f1_score as sk_f1
        val_f1 = sk_f1(all_labels, all_preds, zero_division=0)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['lr'].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best base model weights
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'base_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  ⏹  Early stopping at epoch {epoch+1} (best F1: {best_val_f1:.4f})")
                break

    # Reload best weights
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'base_model.pt'),
                                      weights_only=True))
    print(f"  ✅ Base model trained. Best Val F1: {best_val_f1:.4f}")
    return model, history


def train_lora_model(model, train_loader, val_loader, config, lora_config, device):
    """
    Apply LoRA and fine-tune only the adapter layers + output heads.
    """
    print("\n" + "="*60)
    print("  Phase 2: LoRA Fine-Tuning (Parameter Efficient)")
    print("="*60)

    # Apply LoRA
    model = apply_lora(
        model, r=lora_config.r, lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules
    )
    model = model.to(device)

    # Only optimize trainable (LoRA + head) parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate * 0.5,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs // 2
    )
    loss_fn = MultiTaskLoss(
        cls_weight=config.classification_weight,
        sev_weight=config.severity_weight
    )

    best_val_f1 = 0
    patience_counter = 0
    lora_history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'lr': []}

    lora_epochs = config.epochs // 2  # LoRA typically needs fewer epochs

    for epoch in range(lora_epochs):
        # ── Training ──
        model.train()
        train_losses = []
        for X_batch, y_batch, sev_batch in train_loader:
            X_batch, y_batch, sev_batch = (
                X_batch.to(device), y_batch.to(device), sev_batch.to(device)
            )
            optimizer.zero_grad()
            outputs = model(X_batch)
            losses = loss_fn(outputs, y_batch, sev_batch)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            train_losses.append(losses['total'].item())

        scheduler.step()

        # ── Validation ──
        model.eval()
        val_losses, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for X_batch, y_batch, sev_batch in val_loader:
                X_batch, y_batch, sev_batch = (
                    X_batch.to(device), y_batch.to(device), sev_batch.to(device)
                )
                outputs = model(X_batch)
                losses = loss_fn(outputs, y_batch, sev_batch)
                val_losses.append(losses['total'].item())

                proba = torch.sigmoid(outputs['logits'])
                all_preds.extend((proba >= 0.5).long().cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        from sklearn.metrics import f1_score as sk_f1
        val_f1 = sk_f1(all_labels, all_preds, zero_division=0)

        lora_history['train_loss'].append(np.mean(train_losses))
        lora_history['val_loss'].append(np.mean(val_losses))
        lora_history['val_f1'].append(val_f1)
        lora_history['lr'].append(scheduler.get_last_lr()[0])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{lora_epochs} | "
                  f"Train Loss: {np.mean(train_losses):.4f} | "
                  f"Val Loss: {np.mean(val_losses):.4f} | "
                  f"Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), TRANSFORMER_WEIGHTS_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"  ⏹  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(TRANSFORMER_WEIGHTS_PATH, weights_only=True))
    print(f"  ✅ LoRA fine-tuning complete. Best Val F1: {best_val_f1:.4f}")
    return model, lora_history


def evaluate_model(model, X_test, y_test, device, model_name="FT-Transformer+LoRA"):
    """Full evaluation with enterprise metrics."""
    print(f"\n📊 Evaluating {model_name}...")
    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        y_proba = torch.sigmoid(outputs['logits']).cpu().numpy()
        y_pred = (y_proba >= 0.5).astype(int)
        severity = outputs['severity'].cpu().numpy()

    # Generate comprehensive report
    report = generate_full_report(
        y_test, y_pred, y_proba, model_name,
        save_path=os.path.join(MODEL_DIR, 'evaluation_report.json')
    )

    # Print summary
    metrics = report['classification_metrics']
    nasa = report.get('nasa_asymmetric_score', {})
    cost = report['cost_analysis']

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Results")
    print(f"{'─'*50}")
    print(f"  Accuracy:       {metrics['test_accuracy']:.4f}")
    print(f"  F1 (Macro):     {metrics['test_f1_macro']:.4f}")
    print(f"  F1 (Binary):    {metrics['test_f1_binary']:.4f}")
    print(f"  PR-AUC:         {metrics.get('test_pr_auc', 0):.4f}")
    print(f"  AUC-ROC:        {metrics.get('test_auc_roc', 0):.4f}")
    print(f"  MCC:            {metrics.get('test_mcc', 0):.4f}")
    if nasa:
        print(f"  NASA Score:     {nasa['nasa_score_per_sample']:.4f}")
    print(f"  Net Value:      ${cost['net_value_usd']:,.0f}")
    print(f"{'─'*50}")

    return report, y_pred, y_proba, severity


def generate_shap_explanations(model, X_test, feature_names, device):
    """Generate SHAP explanations for the transformer model."""
    print("\n🧠 Generating SHAP explanations...")
    try:
        import shap

        model.eval()
        X_sample = X_test[:200]  # Sample for speed

        def model_predict(x):
            with torch.no_grad():
                tensor = torch.FloatTensor(x).to(device)
                outputs = model(tensor)
                return torch.sigmoid(outputs['logits']).cpu().numpy()

        # Use KernelExplainer for neural network models
        background = X_test[np.random.choice(len(X_test), 50, replace=False)]
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)

        # Save SHAP values
        np.save(os.path.join(MODEL_DIR, 'shap_values.npy'), shap_values)
        np.save(os.path.join(MODEL_DIR, 'shap_X_sample.npy'), X_sample)

        # Save feature names
        with open(os.path.join(MODEL_DIR, 'shap_feature_names.json'), 'w') as f:
            json.dump(feature_names, f)

        # Generate SHAP plots
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=feature_names, show=False)
        plt.title('SHAP Summary — Feature Impact on Engine Failure Prediction')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'shap_summary.png'),
                   bbox_inches='tight', dpi=150)
        plt.close()

        # Bar plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=feature_names, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance (Mean |SHAP Value|)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'shap_bar.png'),
                   bbox_inches='tight', dpi=150)
        plt.close()

        print("  ✅ SHAP explanations saved")
        return shap_values

    except Exception as e:
        print(f"  ⚠️ SHAP generation warning: {e}")
        return None


def generate_training_plots(base_history, lora_history):
    """Generate training visualization plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss curves
    ax = axes[0]
    ax.plot(base_history['train_loss'], label='Base Train', color='#667eea', alpha=0.7)
    ax.plot(base_history['val_loss'], label='Base Val', color='#667eea', linestyle='--')
    offset = len(base_history['train_loss'])
    lora_x = range(offset, offset + len(lora_history['train_loss']))
    ax.plot(lora_x, lora_history['train_loss'], label='LoRA Train', color='#e74c3c', alpha=0.7)
    ax.plot(lora_x, lora_history['val_loss'], label='LoRA Val', color='#e74c3c', linestyle='--')
    ax.axvline(x=offset, color='gray', linestyle=':', alpha=0.5, label='LoRA Start')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss — Base → LoRA Fine-Tuning')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. F1 Score
    ax = axes[1]
    ax.plot(base_history['val_f1'], label='Base', color='#667eea', linewidth=2)
    ax.plot(lora_x, lora_history['val_f1'], label='LoRA', color='#e74c3c', linewidth=2)
    ax.axvline(x=offset, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 — Base vs LoRA')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Learning rate
    ax = axes[2]
    ax.plot(base_history['lr'], label='Base LR', color='#667eea')
    ax.plot(lora_x, lora_history['lr'], label='LoRA LR', color='#e74c3c')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'transformer_training.png'),
               bbox_inches='tight', dpi=150)
    plt.close()
    print("  📈 Training plots saved")


def main():
    """Main training pipeline."""
    start_time = time.time()
    ensure_dirs()

    # ── Device setup ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──
    df = load_and_preprocess_data()
    (X_train, y_train, X_test, y_test,
     scaler, feature_names, n_features) = prepare_features(df)

    # ── Create DataLoaders ──
    config = TRANSFORMER_CONFIG
    config.n_features = n_features

    train_dataset = SensorDataset(X_train, y_train)
    test_dataset = SensorDataset(X_test, y_test)

    # Split training into train/val (90/10)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_STATE)
    )

    train_loader = DataLoader(train_subset, batch_size=config.batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size,
                           shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # ── MLflow Experiment ──
    mlflow.set_experiment(EXPERIMENT_CONFIG.experiment_name)

    with mlflow.start_run(run_name="FT-Transformer+LoRA"):
        # Log configuration
        mlflow.log_params({
            'n_features': n_features,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_feedforward': config.d_feedforward,
            'dropout': config.dropout,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'lora_r': LORA_CONFIG.r,
            'lora_alpha': LORA_CONFIG.lora_alpha,
            'lora_dropout': LORA_CONFIG.lora_dropout,
            'device': str(device),
        })
        mlflow.set_tags(EXPERIMENT_CONFIG.run_tags)

        # ── Phase 1: Train Base Model ──
        model = FTTransformer(
            n_features=n_features,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
            activation=config.activation
        ).to(device)

        base_stats = get_parameter_stats(model)
        mlflow.log_metrics({
            'base_total_params': base_stats['total_params'],
            'base_trainable_params': base_stats['trainable_params'],
        })

        model, base_history = train_base_model(
            model, train_loader, val_loader, config, device
        )

        # ── Phase 2: LoRA Fine-Tuning ──
        model, lora_history = train_lora_model(
            model, train_loader, val_loader, config, LORA_CONFIG, device
        )

        lora_stats = get_parameter_stats(model)
        mlflow.log_metrics({
            'lora_total_params': lora_stats['total_params'],
            'lora_trainable_params': lora_stats['trainable_params'],
            'lora_frozen_params': lora_stats['frozen_params'],
            'param_reduction_pct': lora_stats['reduction_pct'],
        })

        # ── Evaluation ──
        report, y_pred, y_proba, severity = evaluate_model(
            model, X_test, y_test, device
        )

        # Log metrics to MLflow
        for key, value in report['classification_metrics'].items():
            mlflow.log_metric(key, value)
        if 'nasa_asymmetric_score' in report:
            mlflow.log_metric('nasa_score',
                            report['nasa_asymmetric_score']['nasa_score_per_sample'])

        # ── Training Plots ──
        generate_training_plots(base_history, lora_history)

        # ── SHAP ──
        shap_values = generate_shap_explanations(model, X_test, feature_names, device)

        # ── Save Artifacts ──
        # Save training history
        combined_history = {
            'base': base_history,
            'lora': lora_history,
        }
        with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
            json.dump(combined_history, f, indent=2)

        # Save parameter stats
        param_stats = {
            'base': base_stats,
            'lora': lora_stats,
            'feature_names': feature_names,
            'n_features': n_features,
        }
        with open(os.path.join(MODEL_DIR, 'param_stats.json'), 'w') as f:
            json.dump(param_stats, f, indent=2)

        # Save scaler for inference
        import joblib
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'transformer_scaler.joblib'))

        # Update feature_info.json
        feature_info = {
            'feature_names': feature_names,
            'n_features': n_features,
            'raw_sensor_columns': RAW_SENSOR_COLUMNS,
            'best_model_name': 'FT-Transformer+LoRA',
            'best_f1_score': report['classification_metrics']['test_f1_binary'],
            'best_f1_macro': report['classification_metrics']['test_f1_macro'],
            'best_pr_auc': report['classification_metrics'].get('test_pr_auc', 0),
            'best_auc_roc': report['classification_metrics'].get('test_auc_roc', 0),
            'smote_applied': True,
            'lora_rank': LORA_CONFIG.r,
            'lora_alpha': LORA_CONFIG.lora_alpha,
            'param_reduction_pct': lora_stats['reduction_pct'],
            'total_params': lora_stats['total_params'],
            'trainable_params': lora_stats['trainable_params'],
            'architecture': 'FT-Transformer (Feature Tokenizer Transformer)',
            'training_device': str(device),
        }
        with open(os.path.join(MODEL_DIR, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)

        # Log artifacts to MLflow
        mlflow.log_artifacts(PLOTS_DIR, artifact_path="plots")

        elapsed = time.time() - start_time
        mlflow.log_metric('training_time_seconds', elapsed)

        print(f"\n{'='*60}")
        print(f"  ✅ Training Complete!")
        print(f"{'='*60}")
        print(f"  Total time:         {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Model saved to:     {TRANSFORMER_WEIGHTS_PATH}")
        print(f"  Parameter reduction: {lora_stats['reduction_pct']}%")
        print(f"  Best F1 (Macro):    {report['classification_metrics']['test_f1_macro']:.4f}")
        print(f"  Best PR-AUC:        {report['classification_metrics'].get('test_pr_auc', 0):.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
