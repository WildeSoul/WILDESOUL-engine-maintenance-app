# ============================================================================
# config.py — Centralized Configuration for Predictive Maintenance Pipeline
# ============================================================================
"""
Central configuration module for the PGP Capstone Predictive Maintenance project.
Contains all hyperparameters, paths, thresholds, and LoRA configuration.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_building")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
DEPLOYMENT_DIR = os.path.join(PROJECT_ROOT, "deployment")

RAW_DATA_PATH = os.path.join(DATA_DIR, "engine_data.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
TRANSFORMER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "transformer_model.pt")
LORA_ADAPTER_PATH = os.path.join(MODEL_DIR, "lora_adapter")
FEATURE_INFO_PATH = os.path.join(MODEL_DIR, "feature_info.json")
MODEL_COMPARISON_PATH = os.path.join(MODEL_DIR, "model_comparison.json")
CLASSIFICATION_REPORT_PATH = os.path.join(MODEL_DIR, "classification_report.json")

# ── Column Mappings ──────────────────────────────────────────────────────────
RAW_COLUMN_MAP = {
    'Engine rpm': 'Engine_RPM',
    'Lub oil pressure': 'Lub_Oil_Pressure',
    'Fuel pressure': 'Fuel_Pressure',
    'Coolant pressure': 'Coolant_Pressure',
    'lub oil temp': 'Lub_Oil_Temperature',
    'Coolant temp': 'Coolant_Temperature',
    'Engine Condition': 'Engine_Condition'
}

RAW_SENSOR_COLUMNS = [
    'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure',
    'Coolant_Pressure', 'Lub_Oil_Temperature', 'Coolant_Temperature'
]

TARGET_COLUMN = 'Engine_Condition'

# ── Feature Engineering ──────────────────────────────────────────────────────
IQR_FACTOR = 1.5
HIGH_RPM_PERCENTILE = 0.85

# ── Data Splitting ───────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ── Traditional ML ───────────────────────────────────────────────────────────
CV_FOLDS = 5
RANDOMIZED_SEARCH_ITER = 5
SMOTE_RANDOM_STATE = 42

# ── Quality Gate ─────────────────────────────────────────────────────────────
F1_THRESHOLD = 0.60
AUC_THRESHOLD = 0.65
PRAUC_THRESHOLD = 0.60


@dataclass
class TransformerConfig:
    """FT-Transformer (Feature Tokenizer Transformer) configuration."""
    # Architecture
    n_features: int = 21          # Total features after engineering
    d_model: int = 64             # Embedding dimension per feature token
    n_heads: int = 4              # Number of attention heads
    n_layers: int = 3             # Number of transformer encoder layers
    d_feedforward: int = 128      # FFN hidden dimension
    dropout: float = 0.1         # Dropout rate
    activation: str = "gelu"      # Activation function

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15            # Early stopping patience
    scheduler: str = "cosine"     # LR scheduler type

    # Multi-task weights
    classification_weight: float = 1.0
    severity_weight: float = 0.3


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration for PEFT."""
    r: int = 8                    # LoRA rank (low-rank dimension)
    lora_alpha: int = 32          # Scaling factor (alpha/r = scaling)
    lora_dropout: float = 0.05    # Dropout on LoRA layers
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    bias: str = "none"            # Don't train bias terms
    # Computed property: trainable params reduction
    # With r=8, d_model=64: LoRA params = 2 * r * d_model * n_layers
    # vs Full params = d_model * d_model * n_layers
    # Reduction: ~87.5% fewer trainable params


@dataclass
class ExperimentConfig:
    """MLflow experiment tracking configuration."""
    experiment_name: str = "Engine_Predictive_Maintenance_v2"
    tracking_uri: Optional[str] = None  # Local file-based
    run_tags: dict = field(default_factory=lambda: {
        "project": "PGP_Capstone",
        "model_type": "FT-Transformer+LoRA",
        "dataset": "engine_sensor_19535"
    })


# ── Singleton instances ──────────────────────────────────────────────────────
TRANSFORMER_CONFIG = TransformerConfig()
LORA_CONFIG = LoRAConfig()
EXPERIMENT_CONFIG = ExperimentConfig()


# ── Utility ──────────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all required directories."""
    for d in [DATA_DIR, MODEL_DIR, PLOTS_DIR, DEPLOYMENT_DIR]:
        os.makedirs(d, exist_ok=True)
