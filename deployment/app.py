import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Engine Predictive Maintenance — Control Room", page_icon="", layout="wide", initial_sidebar_state="expanded")

# ── Premium CSS Theme ────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main-header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);padding:2rem 2.5rem;border-radius:16px;margin-bottom:1.5rem;color:#fff;box-shadow:0 8px 32px rgba(0,0,0,.25)}
.main-header h1{color:#fff;font-weight:700;font-size:2rem;margin-bottom:.3rem}
.main-header p{color:#b8b8d4;font-size:.95rem;margin:0}
.metric-card{background:linear-gradient(145deg,#1a1a2e,#16213e);border-radius:14px;padding:1.2rem 1.5rem;text-align:center;color:#fff;box-shadow:0 4px 20px rgba(0,0,0,.15);border:1px solid rgba(255,255,255,.05)}
.metric-card .mv{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-card .ml{font-size:.78rem;color:#8888aa;text-transform:uppercase;letter-spacing:1px;margin-top:.3rem}
.result-safe{background:linear-gradient(135deg,#00b09b,#96c93d);padding:1.5rem 2rem;border-radius:14px;color:#fff;text-align:center;font-size:1.3rem;font-weight:600;animation:pulse-safe 2s ease-in-out infinite}
.result-danger{background:linear-gradient(135deg,#eb3349,#f45c43);padding:1.5rem 2rem;border-radius:14px;color:#fff;text-align:center;font-size:1.3rem;font-weight:600;animation:pulse-danger 1.5s ease-in-out infinite}
@keyframes pulse-safe{0%,100%{box-shadow:0 0 10px rgba(0,176,155,.3)}50%{box-shadow:0 0 25px rgba(0,176,155,.6)}}
@keyframes pulse-danger{0%,100%{box-shadow:0 0 10px rgba(235,51,73,.3)}50%{box-shadow:0 0 25px rgba(235,51,73,.6)}}
.shap-card{background:linear-gradient(145deg,#1a1a2e,#16213e);border-radius:14px;padding:1.5rem;color:#fff;border:1px solid rgba(255,255,255,.05)}
.lora-stat{background:linear-gradient(145deg,#0d1b2a,#1b2838);border-radius:12px;padding:1rem 1.2rem;text-align:center;border:1px solid rgba(102,126,234,.2)}
.lora-stat .num{font-size:2rem;font-weight:700;color:#667eea}
.lora-stat .label{font-size:.75rem;color:#8888aa;text-transform:uppercase;letter-spacing:1px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0c29,#1a1a3e)}
section[data-testid="stSidebar"] label{color:#e0e0ff!important}
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
</style>""", unsafe_allow_html=True)

# ── Helper: Compute Advanced Features Inline ─────────────────────────────────
def compute_features_inline(inp_df):
    """Compute all engineered features for a single input or batch. Uses numpy only (no scipy needed)."""
    df = inp_df.copy()
    sensor_cols = ['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']
    vals = df[sensor_cols].values

    # Time-domain
    df['Sensor_RMS'] = np.sqrt(np.mean(vals**2, axis=1))
    n = vals.shape[1]
    mean_v = np.mean(vals, axis=1, keepdims=True)
    std_v = np.std(vals, axis=1, keepdims=True)
    std_safe = np.where(std_v == 0, 1, std_v)
    z = (vals - mean_v) / std_safe
    df['Sensor_Kurtosis'] = np.mean(z**4, axis=1) - 3
    df['Sensor_Skewness'] = np.mean(z**3, axis=1)
    df['Sensor_PeakToPeak'] = np.ptp(vals, axis=1)
    peak_abs = np.max(np.abs(vals), axis=1)
    rms = df['Sensor_RMS'].values
    df['Sensor_CrestFactor'] = np.divide(peak_abs, rms, out=np.zeros_like(rms), where=rms!=0)
    df['Sensor_Std'] = np.std(vals, axis=1)
    sensor_mean = np.mean(vals, axis=1)
    df['Sensor_CV'] = np.divide(df['Sensor_Std'].values, np.abs(sensor_mean), out=np.zeros_like(sensor_mean), where=sensor_mean!=0)

    # Spectral (FFT using numpy)
    centered = vals - vals.mean(axis=1, keepdims=True)
    fft_result = np.fft.fft(centered, axis=1)
    magnitudes = np.abs(fft_result)
    n_unique = n // 2 + 1
    df['FFT_DC'] = magnitudes[:, 0]
    for i in range(1, n_unique):
        df[f'FFT_Mag_{i}'] = magnitudes[:, i]
    df['Spectral_Energy'] = np.sum(magnitudes[:, 1:n_unique]**2, axis=1)
    freq_bins = np.arange(1, n_unique)
    mag_sl = magnitudes[:, 1:n_unique]
    total_mag = mag_sl.sum(axis=1, keepdims=True)
    total_mag = np.where(total_mag == 0, 1, total_mag)
    df['Spectral_Centroid'] = (mag_sl * freq_bins).sum(axis=1) / total_mag.ravel()
    df['Dominant_Freq_Bin'] = np.argmax(mag_sl, axis=1) + 1

    # Interactions
    df['Temp_Pressure_Ratio'] = df['Lub_Oil_Temperature'] / df['Lub_Oil_Pressure'].replace(0, np.nan)
    df['Temp_Pressure_Ratio'] = df['Temp_Pressure_Ratio'].fillna(0)
    df['Coolant_Efficiency'] = df['Coolant_Pressure'] / df['Coolant_Temperature'].replace(0, np.nan)
    df['Coolant_Efficiency'] = df['Coolant_Efficiency'].fillna(0)
    df['Temp_Differential'] = df['Lub_Oil_Temperature'] - df['Coolant_Temperature']
    df['Thermal_Load'] = (df['Lub_Oil_Temperature'] + df['Coolant_Temperature']) / 2
    df['Pressure_Gradient'] = df['Lub_Oil_Pressure'] - df['Fuel_Pressure']
    df['Total_Pressure'] = df['Lub_Oil_Pressure'] + df['Fuel_Pressure'] + df['Coolant_Pressure']
    rpm_safe = df['Engine_RPM'].replace(0, np.nan)
    df['Oil_Press_per_RPM'] = df['Lub_Oil_Pressure'] / rpm_safe * 1000
    df['Oil_Press_per_RPM'] = df['Oil_Press_per_RPM'].fillna(0)

    # Z-scores (using population stats from training data approx)
    for col in ['Engine_RPM','Lub_Oil_Pressure','Lub_Oil_Temperature']:
        col_mean = df[col].mean() if len(df) > 1 else df[col].iloc[0]
        col_std = df[col].std() if len(df) > 1 else 1
        if col_std == 0: col_std = 1
        df[f'{col}_ZScore'] = (df[col] - col_mean) / col_std

    # High RPM flag
    df['High_RPM_Flag'] = (df['Engine_RPM'] > 1062).astype(int)

    # Clean
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

# ── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_sklearn_model():
    # Try loading from HuggingFace Model Hub first
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id="WILDESOUL/engine-maintenance-model", filename="best_model.joblib")
        return joblib.load(model_path)
    except Exception:
        pass
    # Fallback to local file
    for p in ["best_model.joblib","model_building/best_model.joblib"]:
        if os.path.exists(p): return joblib.load(p)
    return None

@st.cache_resource
def load_transformer_model():
    try:
        import torch
        from transformer_model import FTTransformer, apply_lora
        weights_path = "model_building/transformer_model.pt"
        if not os.path.exists(weights_path): return None, None
        param_stats = load_json("param_stats.json")
        n_feat = param_stats.get('n_features', 31) if param_stats else 31
        model = FTTransformer(n_features=n_feat, d_model=64, n_heads=4, n_layers=3, d_feedforward=128, dropout=0.1)
        model = apply_lora(model, r=8, lora_alpha=32, lora_dropout=0.05)
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
        model.eval()
        scaler = None
        scaler_path = "model_building/transformer_scaler.joblib"
        if os.path.exists(scaler_path): scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        return None, None

@st.cache_data
def load_json(fn):
    for p in [fn, f"model_building/{fn}"]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return None

model = load_sklearn_model()
transformer_model, transformer_scaler = load_transformer_model()
feature_info = load_json("feature_info.json")
model_comparison = load_json("model_comparison.json")
eval_report = load_json("evaluation_report.json")
training_history = load_json("training_history.json")
param_stats = load_json("param_stats.json")

has_transformer = transformer_model is not None
model_type = "FT-Transformer + LoRA" if has_transformer else "Traditional ML Ensemble"

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f'<div class="main-header"><h1>Predictive Engine Maintenance — Control Room</h1><p>AI-powered engine health monitoring | {model_type} | Real-time sensor analysis + SHAP explainability</p></div>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Sensor Inputs")
    st.markdown("---")
    rpm = st.slider("Engine RPM", 0, 5000, 750, step=10)
    lub_oil_pressure = st.slider("Oil Pressure (bar)", 0.0, 10.0, 3.5, step=0.1)
    fuel_pressure = st.slider("Fuel Pressure (bar)", 0.0, 25.0, 6.0, step=0.1)
    coolant_pressure = st.slider("Coolant Pressure (bar)", 0.0, 10.0, 2.0, step=0.1)
    lub_oil_temp = st.slider("Oil Temp (C)", 50.0, 120.0, 78.0, step=0.5)
    coolant_temp = st.slider("Coolant Temp (C)", 50.0, 120.0, 78.0, step=0.5)
    st.markdown("---")
    st.markdown(f"**Model Type:** `{model_type}`")
    if feature_info:
        st.markdown(f"**Best F1:** `{feature_info.get('best_f1_score', feature_info.get('best_f1_macro', 0)):.4f}`")
        if 'param_reduction_pct' in feature_info:
            st.markdown(f"**LoRA Reduction:** `{feature_info['param_reduction_pct']}%`")

# Build input and compute features
inp = pd.DataFrame([{'Engine_RPM':rpm,'Lub_Oil_Pressure':lub_oil_pressure,'Fuel_Pressure':fuel_pressure,'Coolant_Pressure':coolant_pressure,'Lub_Oil_Temperature':lub_oil_temp,'Coolant_Temperature':coolant_temp}])
inp_full = compute_features_inline(inp)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Prediction","Sensor Monitoring","SHAP Explainability","Model Comparison","LoRA Experiments","Fleet Status","Batch Predict"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### Engine Health Prediction")

        # Traditional ML prediction
        sklearn_pred, sklearn_cf = None, None
        if model is not None:
            # sklearn model was trained on 6 raw features only
            inp_sklearn = inp[['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']].copy()
            try:
                sklearn_pred = model.predict(inp_sklearn)[0]
                try: sklearn_cf = model.predict_proba(inp_sklearn)[0][1] * 100
                except: sklearn_cf = 100.0 if sklearn_pred == 1 else 0.0
            except Exception:
                # Fallback: model might expect engineered features
                inp_sklearn['Temp_Pressure_Ratio'] = inp_sklearn['Lub_Oil_Temperature'] / inp_sklearn['Lub_Oil_Pressure'].replace(0, np.nan)
                inp_sklearn['Temp_Pressure_Ratio'] = inp_sklearn['Temp_Pressure_Ratio'].fillna(0)
                inp_sklearn['Coolant_Efficiency'] = inp_sklearn['Coolant_Pressure'] / inp_sklearn['Coolant_Temperature'].replace(0, np.nan)
                inp_sklearn['Coolant_Efficiency'] = inp_sklearn['Coolant_Efficiency'].fillna(0)
                inp_sklearn['High_RPM_Flag'] = (inp_sklearn['Engine_RPM'] > 1062).astype(int)
                sklearn_pred = model.predict(inp_sklearn)[0]
                try: sklearn_cf = model.predict_proba(inp_sklearn)[0][1] * 100
                except: sklearn_cf = 100.0 if sklearn_pred == 1 else 0.0

        # Transformer prediction
        tf_pred, tf_cf, tf_severity = None, None, None
        if has_transformer:
            try:
                import torch
                feat_names = param_stats.get('feature_names', inp_full.columns.tolist()) if param_stats else inp_full.columns.tolist()
                sensor_cols_raw = ['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']
                feat_cols = [c for c in feat_names if c in inp_full.columns and c not in ['Engine_Condition']]
                x_vals = inp_full[feat_cols].values.astype(np.float32)
                if transformer_scaler is not None:
                    x_vals = transformer_scaler.transform(x_vals)
                with torch.no_grad():
                    out = transformer_model(torch.FloatTensor(x_vals))
                    tf_cf = float(torch.sigmoid(out['logits']).item()) * 100
                    tf_pred = 1 if tf_cf >= 50 else 0
                    tf_severity = float(out['severity'].item())
            except Exception as e:
                st.warning(f"Transformer prediction error: {e}")

        # Display primary prediction (prefer transformer if available)
        pred = tf_pred if tf_pred is not None else sklearn_pred
        cf = tf_cf if tf_cf is not None else sklearn_cf

        if pred is not None:
            if pred == 1:
                st.markdown('<div class="result-danger">HIGH RISK - Maintenance Required!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-safe">Engine Operating Normally</div>', unsafe_allow_html=True)

            st.markdown("")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=cf, title={'text':"Failure Risk %",'font':{'size':16}},
                gauge={'axis':{'range':[0,100]},'bar':{'color':"#eb3349" if cf>50 else "#00b09b"},
                'steps':[{'range':[0,30],'color':'#1a3a1a'},{'range':[30,70],'color':'#3a3a1a'},{'range':[70,100],'color':'#3a1a1a'}],
                'threshold':{'line':{'color':"white",'width':3},'thickness':0.8,'value':50}}))
            fig.update_layout(height=250, margin=dict(t=40,b=10,l=30,r=30), paper_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'})
            st.plotly_chart(fig, use_container_width=True)

            # Side-by-side model comparison
            if sklearn_pred is not None and tf_pred is not None:
                st.markdown("#### Model Comparison")
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown(f'<div class="metric-card"><div class="ml">Traditional ML</div><div class="mv">{sklearn_cf:.1f}%</div><div class="ml">{"Faulty" if sklearn_pred==1 else "Normal"}</div></div>', unsafe_allow_html=True)
                with mc2:
                    st.markdown(f'<div class="metric-card"><div class="ml">Transformer+LoRA</div><div class="mv">{tf_cf:.1f}%</div><div class="ml">{"Faulty" if tf_pred==1 else "Normal"}</div></div>', unsafe_allow_html=True)

            if pred == 1:
                st.markdown("**Actions:** Schedule inspection, check oil pressure and temperature")
            else:
                st.markdown("**Status:** All parameters normal. Continue regular maintenance.")
        else:
            st.warning("No model loaded.")

    with c2:
        st.markdown("### Engineered Features")
        # Show top features in a styled table
        feat_display = []
        for col in inp_full.columns:
            if col not in ['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']:
                feat_display.append({'Feature': col, 'Value': f"{inp_full[col].values[0]:.4f}"})
        if feat_display:
            st.dataframe(pd.DataFrame(feat_display), use_container_width=True, hide_index=True, height=400)

        if tf_severity is not None:
            st.markdown("### Severity Score")
            st.markdown(f'<div class="metric-card"><div class="mv">{tf_severity:.3f}</div><div class="ml">Failure Severity (0=None, 1=Critical)</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SENSOR MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Real-time Sensor Gauges")
    sensors = [("Engine RPM",rpm,0,5000,"RPM"),("Oil Pressure",lub_oil_pressure,0,10,"bar"),("Fuel Pressure",fuel_pressure,0,25,"bar"),("Coolant Pressure",coolant_pressure,0,10,"bar"),("Oil Temp",lub_oil_temp,50,120,"C"),("Coolant Temp",coolant_temp,50,120,"C")]
    cols = st.columns(3)
    for i,(nm,vl,lo,hi,un) in enumerate(sensors):
        with cols[i%3]:
            pct = (vl - lo) / (hi - lo) if hi > lo else 0
            status = "Normal" if pct < 0.6 else ("Warning" if pct < 0.85 else "Critical")
            status_color = "#00b09b" if status == "Normal" else ("#ffa500" if status == "Warning" else "#eb3349")
            fg = go.Figure(go.Indicator(mode="gauge+number",value=vl,title={'text':f'{nm} [{status}]','font':{'size':13,'color':status_color}},number={'suffix':f" {un}"},gauge={'axis':{'range':[lo,hi]},'bar':{'color':status_color},'steps':[{'range':[lo,lo+(hi-lo)*0.6],'color':'#1a2a1a'},{'range':[lo+(hi-lo)*0.6,lo+(hi-lo)*0.85],'color':'#2a2a1a'},{'range':[lo+(hi-lo)*0.85,hi],'color':'#2a1a1a'}]}))
            fg.update_layout(height=220,margin=dict(t=40,b=5,l=20,r=20),paper_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'})
            st.plotly_chart(fg, use_container_width=True)

    st.markdown("### Sensor Profile Radar")
    nv = [rpm/5000,lub_oil_pressure/10,fuel_pressure/25,coolant_pressure/10,(lub_oil_temp-50)/70,(coolant_temp-50)/70]
    ct = ['RPM','Oil Press','Fuel Press','Coolant Press','Oil Temp','Coolant Temp']
    fr = go.Figure(go.Scatterpolar(r=nv+[nv[0]],theta=ct+[ct[0]],fill='toself',fillcolor='rgba(102,126,234,0.3)',line=dict(color='#667eea',width=2)))
    fr.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(visible=True,range=[0,1],gridcolor='#333')),height=400,margin=dict(t=30,b=30),paper_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'})
    st.plotly_chart(fr, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### SHAP Explainability — Feature Impact Analysis")
    st.markdown("Understanding **which sensors** drive the model's predictions is critical for operator trust.")

    shap_values_path = "model_building/shap_values.npy"
    shap_features_path = "model_building/shap_feature_names.json"
    shap_summary_img = "model_building/plots/shap_summary.png"
    shap_bar_img = "model_building/plots/shap_bar.png"

    has_shap = os.path.exists(shap_values_path) and os.path.exists(shap_features_path)

    if has_shap:
        shap_vals = np.load(shap_values_path, allow_pickle=True)
        with open(shap_features_path) as f:
            shap_feat_names = json.load(f)

        # Global feature importance
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        top_n = min(15, len(shap_feat_names))

        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            y=[shap_feat_names[i] for i in sorted_idx[:top_n]][::-1],
            x=[mean_abs_shap[i] for i in sorted_idx[:top_n]][::-1],
            orientation='h',
            marker_color=['#667eea' if i < 5 else '#764ba2' if i < 10 else '#444' for i in range(top_n)][::-1]
        ))
        fig_shap.update_layout(title="Global Feature Importance (Mean |SHAP Value|)", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'}, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
        st.plotly_chart(fig_shap, use_container_width=True)

        # Show SHAP images if available
        sc1, sc2 = st.columns(2)
        with sc1:
            if os.path.exists(shap_summary_img):
                st.image(shap_summary_img, caption="SHAP Summary (Beeswarm)", use_container_width=True)
        with sc2:
            if os.path.exists(shap_bar_img):
                st.image(shap_bar_img, caption="SHAP Feature Importance", use_container_width=True)
    else:
        st.info("SHAP data not yet generated. Run `python train_transformer.py` to generate SHAP explanations.")
        st.markdown("""
        **What is SHAP?**

        SHAP (SHapley Additive exPlanations) uses game theory to explain individual predictions:
        - Each feature gets a **SHAP value** representing its contribution to the prediction
        - Positive SHAP = pushes prediction toward **Faulty**
        - Negative SHAP = pushes prediction toward **Normal**
        - The sum of all SHAP values = model output

        **Why it matters for Predictive Maintenance:**
        - Operators need to know *which sensor* triggered the alert
        - Enables targeted maintenance (e.g., "oil temperature is the primary risk factor")
        - Builds trust in the AI system
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Model Performance Comparison")

    if model_comparison and 'results' in model_comparison:
        r = model_comparison['results']
        b = model_comparison.get('best_model', '')
        ns = list(r.keys())
        f1s = [r[n].get('f1_score', 0) for n in ns]
        acs = [r[n].get('auc_roc', 0) for n in ns]

        fb = go.Figure()
        fb.add_trace(go.Bar(name='F1 Score',x=ns,y=f1s,marker_color='#667eea',text=[f'{v:.3f}' for v in f1s],textposition='outside'))
        fb.add_trace(go.Bar(name='AUC-ROC',x=ns,y=acs,marker_color='#764ba2',text=[f'{v:.3f}' for v in acs],textposition='outside'))
        fb.update_layout(barmode='group',height=400,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'},yaxis=dict(gridcolor='#333',range=[0,1.15]))
        st.plotly_chart(fb, use_container_width=True)
        st.success(f"Best Traditional Model: **{b}**")

    # Enterprise metrics from transformer
    if eval_report:
        st.markdown("### Enterprise Evaluation — FT-Transformer + LoRA")
        metrics = eval_report.get('classification_metrics', {})
        nasa = eval_report.get('nasa_asymmetric_score', {})
        cost = eval_report.get('cost_analysis', {})

        em1, em2, em3, em4 = st.columns(4)
        with em1:
            st.markdown(f'<div class="metric-card"><div class="mv">{metrics.get("test_f1_macro",0):.4f}</div><div class="ml">Macro F1</div></div>', unsafe_allow_html=True)
        with em2:
            st.markdown(f'<div class="metric-card"><div class="mv">{metrics.get("test_pr_auc",0):.4f}</div><div class="ml">PR-AUC</div></div>', unsafe_allow_html=True)
        with em3:
            st.markdown(f'<div class="metric-card"><div class="mv">{metrics.get("test_mcc",0):.4f}</div><div class="ml">MCC</div></div>', unsafe_allow_html=True)
        with em4:
            st.markdown(f'<div class="metric-card"><div class="mv">{nasa.get("nasa_score_per_sample",0):.4f}</div><div class="ml">NASA Score</div></div>', unsafe_allow_html=True)

        # Cost analysis
        if cost:
            st.markdown("### Business Impact Analysis")
            bc1, bc2, bc3 = st.columns(3)
            with bc1:
                st.metric("Missed Failures (FN)", cost.get('false_negatives', 0), delta=f"-${cost.get('fn_cost_total',0):,.0f}", delta_color="inverse")
            with bc2:
                st.metric("False Alarms (FP)", cost.get('false_positives', 0), delta=f"-${cost.get('fp_cost_total',0):,.0f}", delta_color="inverse")
            with bc3:
                nv = cost.get('net_value_usd', 0)
                st.metric("Net Value", f"${nv:,.0f}", delta="Positive" if nv > 0 else "Negative")

    if not model_comparison and not eval_report:
        st.info("Run training pipeline to generate model comparison data.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: LORA EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### LoRA — Parameter-Efficient Fine-Tuning")
    st.markdown("**LoRA** (Low-Rank Adaptation) injects small trainable matrices into the transformer's attention layers, reducing trainable parameters by **90%+** while maintaining accuracy.")

    if param_stats:
        lora_s = param_stats.get('lora', {})
        base_s = param_stats.get('base', {})

        # Parameter efficiency stats
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            st.markdown(f'<div class="lora-stat"><div class="num">{lora_s.get("total_params",0):,}</div><div class="label">Total Parameters</div></div>', unsafe_allow_html=True)
        with pc2:
            st.markdown(f'<div class="lora-stat"><div class="num">{lora_s.get("trainable_params",0):,}</div><div class="label">Trainable (LoRA)</div></div>', unsafe_allow_html=True)
        with pc3:
            st.markdown(f'<div class="lora-stat"><div class="num">{lora_s.get("frozen_params",0):,}</div><div class="label">Frozen</div></div>', unsafe_allow_html=True)
        with pc4:
            st.markdown(f'<div class="lora-stat"><div class="num">{lora_s.get("reduction_pct",0)}%</div><div class="label">Reduction</div></div>', unsafe_allow_html=True)

        # Pie chart: trainable vs frozen
        fig_pie = go.Figure(go.Pie(
            labels=['Frozen (Base)', 'LoRA Adapters', 'Output Heads'],
            values=[lora_s.get('frozen_params',0), lora_s.get('lora_params',0), lora_s.get('head_params',0)],
            hole=0.5,
            marker_colors=['#333', '#667eea', '#764ba2']
        ))
        fig_pie.update_layout(title="Parameter Distribution", height=350, paper_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'})
        st.plotly_chart(fig_pie, use_container_width=True)

    # Training history
    if training_history:
        st.markdown("### Training Loss Curves")
        base_h = training_history.get('base', {})
        lora_h = training_history.get('lora', {})

        fig_loss = go.Figure()
        if base_h.get('train_loss'):
            fig_loss.add_trace(go.Scatter(y=base_h['train_loss'], name='Base Train', line=dict(color='#667eea', width=2)))
            fig_loss.add_trace(go.Scatter(y=base_h['val_loss'], name='Base Val', line=dict(color='#667eea', dash='dash', width=2)))
        if lora_h.get('train_loss'):
            offset = len(base_h.get('train_loss', []))
            x_lora = list(range(offset, offset + len(lora_h['train_loss'])))
            fig_loss.add_trace(go.Scatter(x=x_lora, y=lora_h['train_loss'], name='LoRA Train', line=dict(color='#e74c3c', width=2)))
            fig_loss.add_trace(go.Scatter(x=x_lora, y=lora_h['val_loss'], name='LoRA Val', line=dict(color='#e74c3c', dash='dash', width=2)))
            fig_loss.add_vline(x=offset, line_dash="dot", line_color="gray", annotation_text="LoRA Start")

        fig_loss.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'}, xaxis=dict(title='Epoch', gridcolor='#333'), yaxis=dict(title='Loss', gridcolor='#333'))
        st.plotly_chart(fig_loss, use_container_width=True)

        # F1 curve
        if base_h.get('val_f1') or lora_h.get('val_f1'):
            fig_f1 = go.Figure()
            if base_h.get('val_f1'):
                fig_f1.add_trace(go.Scatter(y=base_h['val_f1'], name='Base F1', line=dict(color='#667eea', width=2)))
            if lora_h.get('val_f1'):
                offset = len(base_h.get('val_f1', []))
                x_lora = list(range(offset, offset + len(lora_h['val_f1'])))
                fig_f1.add_trace(go.Scatter(x=x_lora, y=lora_h['val_f1'], name='LoRA F1', line=dict(color='#e74c3c', width=2)))
            fig_f1.update_layout(title="Validation F1 Score", height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'}, xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'))
            st.plotly_chart(fig_f1, use_container_width=True)

    if not param_stats and not training_history:
        st.info("Run `python train_transformer.py` to generate LoRA experiment data.")
        st.markdown("""
        ### How LoRA Works

        **Standard Fine-Tuning:** Update ALL model weights (100% trainable)

        **LoRA Fine-Tuning:** Freeze base model, inject small low-rank matrices into attention layers

        ```
        Original: y = Wx
        LoRA:     y = Wx + (alpha/r)(BAx)
        Where:    A is (r x d), B is (d x r), r << d
        ```

        **Benefits:**
        - ~90% fewer trainable parameters
        - 3-5x faster training
        - Much lower GPU memory usage
        - Comparable accuracy to full fine-tuning
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: FLEET STATUS
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Fleet Health Dashboard")
    st.markdown("Upload a CSV with sensor readings from multiple engines to assess fleet-wide health status.")

    fleet_file = st.file_uploader("Upload Fleet Sensor Data (CSV)", type=['csv'], key="fleet")
    if fleet_file and model:
        fleet_df = pd.read_csv(fleet_file)
        fleet_df.rename(columns={'Engine rpm':'Engine_RPM','Lub oil pressure':'Lub_Oil_Pressure','Fuel pressure':'Fuel_Pressure','Coolant pressure':'Coolant_Pressure','lub oil temp':'Lub_Oil_Temperature','Coolant temp':'Coolant_Temperature','Engine Condition':'Engine_Condition'}, inplace=True)
        req = ['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']

        if all(c in fleet_df.columns for c in req):
            bi = fleet_df[req].copy()
            bi_feat = compute_features_inline(bi)

            # Sklearn prediction (model trained on 6 raw features)
            ps = model.predict(bi)
            try: pb = model.predict_proba(bi)[:,1]
            except: pb = ps.astype(float)

            fleet_df['Risk_Score'] = (pb * 100).round(1)
            fleet_df['Status'] = ['Critical' if p > 70 else 'Warning' if p > 30 else 'Normal' for p in fleet_df['Risk_Score']]
            fleet_df['Priority'] = fleet_df['Risk_Score'].rank(ascending=False).astype(int)

            # Summary
            fc1, fc2 = st.columns([1, 1])
            with fc1:
                status_counts = fleet_df['Status'].value_counts()
                fig_fleet = go.Figure(go.Pie(
                    labels=status_counts.index.tolist(),
                    values=status_counts.values.tolist(),
                    hole=0.5,
                    marker_colors=['#eb3349' if s=='Critical' else '#ffa500' if s=='Warning' else '#00b09b' for s in status_counts.index]
                ))
                fig_fleet.update_layout(title="Fleet Health Summary", height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'})
                st.plotly_chart(fig_fleet, use_container_width=True)

            with fc2:
                st.metric("Total Engines", len(fleet_df))
                st.metric("Critical", len(fleet_df[fleet_df['Status']=='Critical']))
                st.metric("Warning", len(fleet_df[fleet_df['Status']=='Warning']))
                st.metric("Normal", len(fleet_df[fleet_df['Status']=='Normal']))

            # Ranked table
            st.markdown("### Engines Ranked by Risk (Highest First)")
            display_df = fleet_df.sort_values('Risk_Score', ascending=False)[['Priority'] + req + ['Risk_Score', 'Status']].head(50)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.download_button("Download Fleet Report", fleet_df.to_csv(index=False), "fleet_report.csv", "text/csv")
        else:
            st.error(f"CSV must contain columns: {req}")
    else:
        st.info("Upload a CSV file to assess fleet health. Required columns: Engine_RPM, Lub_Oil_Pressure, Fuel_Pressure, Coolant_Pressure, Lub_Oil_Temperature, Coolant_Temperature")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: BATCH PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### Batch Prediction via CSV")
    st.markdown("Upload CSV with: Engine_RPM, Lub_Oil_Pressure, Fuel_Pressure, Coolant_Pressure, Lub_Oil_Temperature, Coolant_Temperature")
    up = st.file_uploader("Upload CSV", type=['csv'], key="batch")
    if up and model:
        df = pd.read_csv(up)
        df.rename(columns={'Engine rpm':'Engine_RPM','Lub oil pressure':'Lub_Oil_Pressure','Fuel pressure':'Fuel_Pressure','Coolant pressure':'Coolant_Pressure','lub oil temp':'Lub_Oil_Temperature','Coolant temp':'Coolant_Temperature'}, inplace=True)
        req = ['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']
        if all(c in df.columns for c in req):
            bi = df[req].copy()
            ps = model.predict(bi)
            try: pb = model.predict_proba(bi)[:,1]
            except: pb = ps.astype(float)
            df['Prediction'] = ['Faulty' if p==1 else 'Normal' for p in ps]
            df['Risk'] = [f'{p*100:.1f}%' for p in pb]

            ca, cb = st.columns(2)
            with ca:
                st.metric("Total", len(ps))
                st.metric("Normal", sum(1 for p in ps if p==0))
                st.metric("Faulty", sum(1 for p in ps if p==1))
            with cb:
                fp = go.Figure(go.Pie(labels=['Normal','Faulty'],values=[sum(1 for p in ps if p==0),sum(1 for p in ps if p==1)],marker_colors=['#00b09b','#eb3349'],hole=0.5))
                fp.update_layout(height=300,paper_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'},showlegend=False)
                st.plotly_chart(fp, use_container_width=True)
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button("Download Results",df.to_csv(index=False),"predictions.csv","text/csv")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="text-align:center;color:#666;font-size:.8rem;padding:1rem"><b>Predictive Maintenance</b> | FT-Transformer + LoRA | SHAP Explainability | MLflow | HuggingFace<br>2026 WildeSoul</div>', unsafe_allow_html=True)
