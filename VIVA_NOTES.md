# Viva Preparation — Key Talking Points

## 1. Why FT-Transformer over LSTM?

**Answer:** "I chose the FT-Transformer (Feature Tokenizer Transformer) architecture because:
- **Self-attention** enables the model to learn which sensor combinations matter most, rather than relying on sequential order (LSTMs assume temporal ordering, but our data is tabular sensor snapshots)
- **Parallelization** — Transformers process all features simultaneously, unlike LSTMs which process sequentially, making training 3-5x faster
- **Interpretability** — The attention weights show feature-to-feature interactions, complementing SHAP explanations
- **Published research** — Gorishniy et al. (2021) showed FT-Transformer outperforms GBDTs on many tabular benchmarks"

## 2. Why LoRA over Full Fine-Tuning?

**Answer:** "LoRA (Low-Rank Adaptation) demonstrates parameter efficiency:
- My model has ~115K total parameters, but with LoRA only ~10K are trainable — a **91% reduction**
- LoRA injects small low-rank matrices (rank=8) into the Q and V attention projections: `W' = W + (alpha/r)(BA)x`
- The base transformer weights are frozen — only the LoRA adapters and output heads are trained
- **Business value:** In production, you can deploy one base model and swap LoRA adapters for different engine types, reducing storage by 10x
- Training is 3-5x faster with 60%+ less GPU memory, making it viable on consumer GPUs (RTX 4060)"

## 3. Why NASA Asymmetric Scoring?

**Answer:** "Standard metrics like accuracy treat false positives and false negatives equally. But in predictive maintenance:
- **False Negative** (missed failure): Causes catastrophic engine damage (~$500K)
- **False Positive** (false alarm): Only causes an unnecessary inspection (~$5K)
- The NASA scoring function penalizes late predictions (missed failures) **3x more** than early predictions
- Formula: `s = exp(d/10) - 1` for late vs `s = exp(-d/13) - 1` for early
- This aligns the ML metric with actual business risk"

## 4. Why FFT/Spectral Features?

**Answer:** "I computed the Discrete Fourier Transform of each engine's 6-sensor profile:
- Treating the 6 sensor readings as a discrete signal, the DFT decomposes it into frequency components
- This reveals the 'spectral signature' of the engine state — different failure modes produce distinct frequency patterns
- For example: a faulty engine might have higher energy in the low-frequency bins (slow degradation) vs high-frequency (vibration)
- I also compute spectral energy (total signal power) and spectral centroid (center of mass of the spectrum)
- Combined with time-domain features (RMS, Kurtosis, Crest Factor), this creates a rich feature representation"

## 5. Feature Engineering Depth

**Answer:** "I engineered 25 features from 6 raw sensors across three categories:
1. **Time-domain (7):** RMS, Kurtosis, Skewness, Peak-to-Peak, Crest Factor, Std, CV
2. **Spectral (8):** FFT magnitudes (3 bins), spectral energy, centroid, dominant frequency
3. **Domain-specific (10):** Temperature-pressure ratios, thermal load, pressure gradients, Z-score anomaly indicators
- Total: 31 features (6 raw + 25 engineered)
- Each feature has physical meaning — Kurtosis detects 'peaky' sensor distributions indicating degradation, Crest Factor identifies shock/spike events"

## 6. SHAP Explainability

**Answer:** "Industrial operators will not trust a black-box model. I implemented SHAP (SHapley Additive exPlanations):
- Uses game theory to attribute each prediction to individual feature contributions
- For each prediction, operators can see exactly which sensor is driving the failure alert
- Example: 'The model flagged this engine because Oil Temperature (SHAP=+0.35) and Coolant Efficiency (SHAP=+0.22) are the primary risk factors'
- This enables **targeted maintenance** — inspect the specific subsystem, not the entire engine
- SHAP is integrated directly into the Streamlit dashboard for real-time explanations"

## 7. Enterprise Evaluation Strategy

**Answer:** "I evaluate beyond standard accuracy using 4 metric categories:
1. **Macro F1 & PR-AUC** — Handle class imbalance (failures are rare events, ~40% of data)
2. **MCC (Matthews Correlation Coefficient)** — Most balanced metric, robust to imbalanced datasets
3. **NASA Asymmetric Scoring** — Business-aware metric penalizing missed failures 3x
4. **Cost-Weighted Analysis** — Translates model performance into dollar values ($500K per missed failure vs $5K per false alarm)
- The cost analysis shows the model's **net value** to the business"

## 8. Architecture Diagram for Slides

```
Input (6 sensors) -> Advanced Feature Engineering (31 features)
    |
    v
Feature Tokenizer (each feature -> 64-dim embedding)
    |
    v
[CLS] + 31 Feature Tokens -> Transformer Encoder (3 layers, 4 heads)
    |                              |
    |                    [LoRA on Q, V matrices]
    |                    (rank=8, alpha=32)
    |                    91% parameter reduction
    v
[CLS] representation (64-dim)
    |
    +---> Classification Head -> Normal/Faulty
    +---> Severity Head -> Failure Severity [0,1]
    |
    v
SHAP Explanations -> Which sensors drive each prediction
```

## 9. Tech Stack Summary

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Transformer | Custom FT-Transformer |
| PEFT | LoRA (custom implementation) |
| Feature Engineering | scipy (FFT, PSD), numpy |
| Explainability | SHAP (KernelExplainer) |
| Experiment Tracking | MLflow |
| Deployment | Streamlit on HuggingFace Spaces |
| CI/CD | GitHub Actions |
| GPU Training | RTX 4060 (local) |
