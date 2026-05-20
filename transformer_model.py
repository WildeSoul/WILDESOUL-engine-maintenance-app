# ============================================================================
# transformer_model.py — FT-Transformer with LoRA for Tabular Sensor Data
# ============================================================================
"""
Feature Tokenizer Transformer (FT-Transformer) with LoRA fine-tuning
for the PGP Capstone Predictive Maintenance project.

Architecture (Gorishniy et al., 2021 — adapted):
    Input Features → Feature Tokenizer → [CLS] + Token Embeddings
    → Transformer Encoder (with LoRA on Q, V projections)
    → Multi-Task Heads (Classification + Severity)

LoRA (Low-Rank Adaptation) — Hu et al., 2021:
    Injects low-rank matrices into attention projections:
    W' = W + α/r · (B × A)  where A ∈ R^(r×d), B ∈ R^(d×r)
    This reduces trainable parameters by ~87-95% while maintaining accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA Layer Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Linear layer with LoRA (Low-Rank Adaptation).

    Replaces: y = Wx + b
    With:     y = Wx + (α/r)(BAx) + b

    Where:
        W ∈ R^(out×in)  — frozen pretrained weights
        A ∈ R^(r×in)    — trainable low-rank down-projection
        B ∈ R^(out×r)   — trainable low-rank up-projection
        r               — rank (much smaller than min(in, out))
        α               — scaling factor

    Parameter reduction:
        Full: in × out = d² parameters
        LoRA: r × (in + out) = 2rd parameters
        Reduction: 1 - 2r/d ≈ 87.5% for r=8, d=64
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8,
                 lora_alpha: int = 32, lora_dropout: float = 0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original path (frozen)
        result = self.original_linear(x)

        # LoRA path (trainable): x → dropout → A → B → scale
        lora_out = self.lora_dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # x @ A^T
        lora_out = F.linear(lora_out, self.lora_B)  # (xA^T) @ B^T
        result = result + self.scaling * lora_out

        return result

    def extra_repr(self) -> str:
        return (f"in={self.original_linear.in_features}, "
                f"out={self.original_linear.out_features}, "
                f"r={self.r}, alpha={self.lora_alpha}, "
                f"scaling={self.scaling:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# FT-Transformer Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    """
    Converts each numeric feature into a d_model-dimensional token embedding.

    Each feature gets its own learned linear projection, allowing the model
    to learn feature-specific representations. A [CLS] token is prepended
    for classification aggregation.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Each feature gets its own embedding layer
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])

        # Learnable [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Feature-level bias (learnable position-like encoding)
        self.feature_bias = nn.Parameter(torch.zeros(1, n_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_features) — raw feature values

        Returns:
            (batch_size, n_features + 1, d_model) — [CLS] + feature tokens
        """
        batch_size = x.size(0)

        # Embed each feature independently
        tokens = []
        for i, embed_layer in enumerate(self.feature_embeddings):
            feat_val = x[:, i:i+1]  # (batch, 1)
            tokens.append(embed_layer(feat_val))  # (batch, d_model)

        # Stack into token sequence: (batch, n_features, d_model)
        token_seq = torch.stack(tokens, dim=1)

        # Add feature-level bias
        token_seq = token_seq + self.feature_bias

        # Prepend [CLS] token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        token_seq = torch.cat([cls, token_seq], dim=1)  # (batch, n_features+1, d_model)

        return token_seq


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with named Q, K, V projections
    (required for LoRA injection into specific projection matrices).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Named projections for LoRA targeting
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm architecture."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual for attention
        x = x + self.attn(self.norm1(x))
        # Pre-norm + residual for FFN
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular sensor data.

    Architecture:
        [Features] → FeatureTokenizer → TransformerEncoder → Multi-Task Heads

    Multi-task outputs:
        1. Classification Head: Binary engine condition (Normal/Faulty)
        2. Severity Head: Continuous failure severity score [0, 1]
    """

    def __init__(self, n_features: int = 21, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3,
                 d_feedforward: int = 128, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.n_layers = n_layers

        # Feature tokenization
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        # Transformer encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_feedforward, dropout, activation)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Multi-task output heads
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, n_features) — input feature vector

        Returns:
            Dictionary with 'logits' (classification) and 'severity' (regression)
        """
        # Tokenize features: (batch, n_features+1, d_model)
        tokens = self.tokenizer(x)

        # Pass through transformer encoder
        for layer in self.encoder_layers:
            tokens = layer(tokens)

        # Final normalization
        tokens = self.final_norm(tokens)

        # Use [CLS] token for prediction
        cls_repr = tokens[:, 0, :]  # (batch, d_model)

        # Multi-task predictions
        logits = self.classification_head(cls_repr)        # (batch, 1)
        severity = self.severity_head(cls_repr)            # (batch, 1)

        return {
            'logits': logits.squeeze(-1),      # (batch,) — raw logits for BCEWithLogitsLoss
            'severity': severity.squeeze(-1),  # (batch,) — [0, 1] severity score
            'cls_embedding': cls_repr          # (batch, d_model) — for SHAP
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference mode prediction with probabilities."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            proba = torch.sigmoid(outputs['logits'])
            pred = (proba >= 0.5).long()
            return {
                'prediction': pred,
                'probability': proba,
                'severity': outputs['severity'],
            }


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA Injection
# ═══════════════════════════════════════════════════════════════════════════════

def apply_lora(model: FTTransformer, r: int = 8, lora_alpha: int = 32,
               lora_dropout: float = 0.05,
               target_modules: List[str] = None) -> FTTransformer:
    """
    Inject LoRA adapters into the FT-Transformer's attention projections.

    This freezes all base model parameters and only makes LoRA matrices
    trainable, achieving ~87-95% parameter reduction.

    Args:
        model: FTTransformer instance (pretrained or initialized)
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout on LoRA layers
        target_modules: Which projections to inject (default: ['q_proj', 'v_proj'])

    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']

    # Step 1: Freeze ALL base model parameters
    for param in model.parameters():
        param.requires_grad = False

    lora_count = 0

    # Step 2: Inject LoRA into target attention modules
    for layer in model.encoder_layers:
        attn = layer.attn
        for module_name in target_modules:
            if hasattr(attn, module_name):
                original_linear = getattr(attn, module_name)
                lora_layer = LoRALinear(
                    original_linear, r=r,
                    lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
                setattr(attn, module_name, lora_layer)
                lora_count += 1

    # Step 3: Unfreeze output heads (always trainable)
    for param in model.classification_head.parameters():
        param.requires_grad = True
    for param in model.severity_head.parameters():
        param.requires_grad = True

    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    reduction_pct = (1 - trainable_params / total_params) * 100

    print(f"\n{'='*60}")
    print(f"  LoRA Injection Summary")
    print(f"{'='*60}")
    print(f"  LoRA layers injected: {lora_count}")
    print(f"  LoRA rank (r):        {r}")
    print(f"  LoRA alpha:           {lora_alpha}")
    print(f"  Scaling (a/r):        {lora_alpha/r:.1f}")
    print(f"{'─'*60}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters:    {frozen_params:,}")
    print(f"  Parameter reduction:  {reduction_pct:.1f}%")
    print(f"{'='*60}\n")

    return model


def get_parameter_stats(model: nn.Module) -> Dict[str, int]:
    """Get detailed parameter statistics for reporting."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    # LoRA-specific params
    lora_params = 0
    head_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora_' in name:
                lora_params += param.numel()
            elif 'head' in name:
                head_params += param.numel()

    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': frozen,
        'lora_params': lora_params,
        'head_params': head_params,
        'reduction_pct': round((1 - trainable / total) * 100, 1) if total > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset and DataLoader
# ═══════════════════════════════════════════════════════════════════════════════

class SensorDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for sensor feature data."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 severity: Optional[np.ndarray] = None):
        """
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (n_samples,)
            severity: Optional severity scores (n_samples,) in [0,1]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        # If no severity labels, use binary labels as proxy
        self.severity = torch.FloatTensor(
            severity if severity is not None else y.astype(float)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.severity[idx]


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.

    Loss = w_cls * BCEWithLogitsLoss(logits, labels)
         + w_sev * MSELoss(severity_pred, severity_true)
    """

    def __init__(self, cls_weight: float = 1.0, sev_weight: float = 0.3):
        super().__init__()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.sev_loss = nn.MSELoss()
        self.cls_weight = cls_weight
        self.sev_weight = sev_weight

    def forward(self, outputs: Dict[str, torch.Tensor],
                labels: torch.Tensor,
                severity: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_cls = self.cls_loss(outputs['logits'], labels)
        loss_sev = self.sev_loss(outputs['severity'], severity)
        total = self.cls_weight * loss_cls + self.sev_weight * loss_sev

        return {
            'total': total,
            'classification': loss_cls,
            'severity': loss_sev,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing FT-Transformer with LoRA...\n")

    # Create model
    n_features = 21
    model = FTTransformer(
        n_features=n_features,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_feedforward=128,
        dropout=0.1
    )

    # Print base model stats
    base_stats = get_parameter_stats(model)
    print(f"Base model: {base_stats['total_params']:,} params "
          f"(all {base_stats['trainable_params']:,} trainable)")

    # Apply LoRA
    model = apply_lora(model, r=8, lora_alpha=32, lora_dropout=0.05)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, n_features)
    outputs = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Severity shape: {outputs['severity'].shape}")
    print(f"CLS embedding shape: {outputs['cls_embedding'].shape}")

    # Test prediction
    preds = model.predict(x)
    print(f"\nPredictions: {preds['prediction'][:5]}")
    print(f"Probabilities: {preds['probability'][:5].numpy().round(3)}")
    print(f"Severity: {preds['severity'][:5].numpy().round(3)}")

    # Test loss
    loss_fn = MultiTaskLoss()
    y = torch.randint(0, 2, (batch_size,)).float()
    sev = torch.rand(batch_size)
    losses = loss_fn(model(x), y, sev)
    print(f"\nLoss - Total: {losses['total']:.4f}, "
          f"Cls: {losses['classification']:.4f}, "
          f"Sev: {losses['severity']:.4f}")

    print("\n✅ FT-Transformer + LoRA test passed!")
