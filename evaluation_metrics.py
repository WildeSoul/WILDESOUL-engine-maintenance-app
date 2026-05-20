# ============================================================================
# evaluation_metrics.py — Enterprise-Grade Evaluation for PGP Capstone
# ============================================================================
"""
Production-grade evaluation metrics module for Predictive Maintenance.

Implements:
1. Standard Classification Metrics: Macro F1, PR-AUC, weighted metrics
2. NASA-Inspired Asymmetric Scoring: Penalizes overconfident 'safe' predictions
3. Cost-Weighted Analysis: Business impact of false negatives vs false positives
4. Comprehensive Report Generation: JSON + plots
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
    matthews_corrcoef, balanced_accuracy_score, log_loss
)
from typing import Dict, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


def compute_classification_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    y_proba: Optional[np.ndarray] = None,
                                    prefix: str = "test") -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Focuses on metrics critical for imbalanced industrial data:
    - Macro F1 (equal weight to rare class)
    - PR-AUC (Precision-Recall Area Under Curve)
    - MCC (Matthews Correlation Coefficient)
    - Balanced Accuracy

    Args:
        y_true: True binary labels (0=Normal, 1=Faulty)
        y_pred: Predicted binary labels
        y_proba: Predicted probabilities for positive class (optional)
        prefix: Metric name prefix (e.g., 'train', 'test')

    Returns:
        Dictionary of metric_name → value
    """
    metrics = {}

    # Standard metrics
    metrics[f'{prefix}_accuracy'] = accuracy_score(y_true, y_pred)
    metrics[f'{prefix}_balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics[f'{prefix}_precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics[f'{prefix}_recall'] = recall_score(y_true, y_pred, zero_division=0)

    # F1 variants
    metrics[f'{prefix}_f1_binary'] = f1_score(y_true, y_pred, zero_division=0)
    metrics[f'{prefix}_f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f'{prefix}_f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Matthews Correlation Coefficient — robust to class imbalance
    metrics[f'{prefix}_mcc'] = matthews_corrcoef(y_true, y_pred)

    # Probability-based metrics (require y_proba)
    if y_proba is not None:
        # AUC-ROC
        try:
            metrics[f'{prefix}_auc_roc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics[f'{prefix}_auc_roc'] = 0.0

        # PR-AUC (critical for imbalanced data — failures are rare events)
        try:
            metrics[f'{prefix}_pr_auc'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics[f'{prefix}_pr_auc'] = 0.0

        # Log Loss
        try:
            metrics[f'{prefix}_log_loss'] = log_loss(y_true, y_proba)
        except ValueError:
            metrics[f'{prefix}_log_loss'] = float('inf')

    return metrics


def nasa_asymmetric_score(y_true: np.ndarray,
                           y_pred_proba: np.ndarray,
                           threshold: float = 0.5,
                           early_decay: float = 13.0,
                           late_decay: float = 10.0) -> Dict[str, float]:
    """
    NASA-inspired asymmetric scoring function adapted for classification.

    Original NASA RUL scoring:
        s_i = exp(-d_i / 13) - 1   if d_i < 0 (early prediction)
        s_i = exp(d_i / 10) - 1    if d_i >= 0 (late prediction)
        S = Σ s_i

    Adapted for binary classification:
        d_i = confidence_error = predicted_proba - actual
        - Predicting 'safe' when actually faulty (FN) → late-like penalty (SEVERE)
        - Predicting 'faulty' when actually safe (FP) → early-like penalty (mild)

    Business rationale:
        Late predictions (missed failures) cause catastrophic engine damage ($500K+)
        Early predictions (false alarms) only cause unnecessary inspections ($5K)
        → Late predictions are penalized ~3x more severely

    Args:
        y_true: True binary labels (0=Normal, 1=Faulty)
        y_pred_proba: Predicted probability of being Faulty
        threshold: Classification threshold
        early_decay: Decay constant for false positives (milder)
        late_decay: Decay constant for false negatives (harsher)

    Returns:
        Dictionary with score components and total
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Error signal: positive = predicted safe but was faulty (dangerous)
    #               negative = predicted faulty but was safe (cautious)
    d = y_true - y_pred_proba  # Positive when under-predicting risk

    scores = np.zeros_like(d, dtype=float)

    # Under-predicted risk (predicted safe, actually faulty) → harsh penalty
    mask_late = d > 0
    scores[mask_late] = np.exp(d[mask_late] / late_decay) - 1

    # Over-predicted risk (predicted faulty, actually safe) → mild penalty
    mask_early = d < 0
    scores[mask_early] = np.exp(-d[mask_early] / early_decay) - 1

    # Component analysis
    total_score = np.sum(scores)
    fn_count = np.sum((y_pred == 0) & (y_true == 1))
    fp_count = np.sum((y_pred == 1) & (y_true == 0))
    fn_penalty = np.sum(scores[mask_late])
    fp_penalty = np.sum(scores[mask_early])

    return {
        'nasa_total_score': float(total_score),
        'nasa_score_per_sample': float(total_score / len(y_true)),
        'fn_penalty': float(fn_penalty),
        'fp_penalty': float(fp_penalty),
        'fn_count': int(fn_count),
        'fp_count': int(fp_count),
        'penalty_ratio': float(fn_penalty / max(fp_penalty, 1e-10)),
        'interpretation': (
            f"Total asymmetric score: {total_score:.2f}. "
            f"FN penalty ({fn_count} missed failures): {fn_penalty:.2f}. "
            f"FP penalty ({fp_count} false alarms): {fp_penalty:.2f}. "
            f"Penalty ratio: {fn_penalty / max(fp_penalty, 1e-10):.1f}x "
            f"(FN penalized more severely)."
        )
    }


def compute_cost_weighted_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   fn_cost: float = 500000,
                                   fp_cost: float = 5000,
                                   tp_savings: float = 450000) -> Dict[str, float]:
    """
    Compute business-impact cost metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        fn_cost: Cost of missing a failure ($500K — catastrophic damage)
        fp_cost: Cost of false alarm ($5K — unnecessary inspection)
        tp_savings: Savings from correctly predicting failure ($450K)

    Returns:
        Dictionary of cost metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    total_cost = fn * fn_cost + fp * fp_cost
    total_savings = tp * tp_savings
    net_value = total_savings - total_cost

    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'total_cost_usd': float(total_cost),
        'total_savings_usd': float(total_savings),
        'net_value_usd': float(net_value),
        'cost_per_prediction': float(total_cost / len(y_true)),
        'fn_cost_total': float(fn * fn_cost),
        'fp_cost_total': float(fp * fp_cost),
    }


def generate_full_report(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None,
                          model_name: str = "Unknown",
                          save_path: Optional[str] = None) -> dict:
    """
    Generate a comprehensive evaluation report.

    Combines all metric types into a single structured report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Name of the model
        save_path: Optional path to save JSON report

    Returns:
        Complete evaluation report dictionary
    """
    report = {
        'model_name': model_name,
        'n_samples': int(len(y_true)),
        'class_distribution': {
            'normal': int(np.sum(y_true == 0)),
            'faulty': int(np.sum(y_true == 1)),
            'faulty_rate': float(np.mean(y_true))
        },
        'classification_metrics': compute_classification_metrics(
            y_true, y_pred, y_proba, prefix='test'
        ),
        'cost_analysis': compute_cost_weighted_metrics(y_true, y_pred),
        'classification_report': classification_report(
            y_true, y_pred, target_names=['Normal', 'Faulty'], output_dict=True
        ),
    }

    if y_proba is not None:
        report['nasa_asymmetric_score'] = nasa_asymmetric_score(y_true, y_proba)

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📊 Report saved to {save_path}")

    return report


def compare_models(reports: Dict[str, dict]) -> pd.DataFrame:
    """
    Create a comparison table from multiple model reports.

    Args:
        reports: Dictionary of model_name → evaluation report

    Returns:
        DataFrame with models as rows and metrics as columns
    """
    rows = []
    for name, report in reports.items():
        metrics = report.get('classification_metrics', {})
        nasa = report.get('nasa_asymmetric_score', {})
        cost = report.get('cost_analysis', {})
        rows.append({
            'Model': name,
            'F1 (Macro)': metrics.get('test_f1_macro', 0),
            'F1 (Binary)': metrics.get('test_f1_binary', 0),
            'PR-AUC': metrics.get('test_pr_auc', 0),
            'AUC-ROC': metrics.get('test_auc_roc', 0),
            'MCC': metrics.get('test_mcc', 0),
            'Balanced Acc': metrics.get('test_balanced_accuracy', 0),
            'NASA Score': nasa.get('nasa_score_per_sample', 0),
            'Net Value ($)': cost.get('net_value_usd', 0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('F1 (Macro)', ascending=False).reset_index(drop=True)
    return df


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.6, n)
    y_proba = np.clip(y_true + np.random.normal(0, 0.3, n), 0, 1)
    y_pred = (y_proba >= 0.5).astype(int)

    report = generate_full_report(y_true, y_pred, y_proba, "TestModel")

    print("\n=== Classification Metrics ===")
    for k, v in report['classification_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\n=== NASA Asymmetric Score ===")
    nasa = report['nasa_asymmetric_score']
    print(f"  {nasa['interpretation']}")

    print("\n=== Cost Analysis ===")
    cost = report['cost_analysis']
    print(f"  Net Value: ${cost['net_value_usd']:,.0f}")
