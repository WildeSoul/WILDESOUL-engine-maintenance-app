import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
<<<<<<< HEAD
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem; color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.25); }
.main-header h1 { color: white; font-weight: 700; font-size: 2rem; margin-bottom: 0.3rem; }
.main-header p { color: #b8b8d4; font-size: 0.95rem; margin: 0; }
.metric-card { background: linear-gradient(145deg, #1a1a2e, #16213e); border-radius: 14px; padding: 1.2rem 1.5rem; text-align: center; color: white; box-shadow: 0 4px 20px rgba(0,0,0,0.15); border: 1px solid rgba(255,255,255,0.05); }
.metric-card .metric-value { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-card .metric-label { font-size: 0.78rem; color: #8888aa; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }
.result-safe { background: linear-gradient(135deg, #00b09b, #96c93d); padding: 1.5rem 2rem; border-radius: 14px; color: white; text-align: center; font-size: 1.3rem; font-weight: 600; box-shadow: 0 6px 25px rgba(0,176,155,0.3); }
.result-danger { background: linear-gradient(135deg, #eb3349, #f45c43); padding: 1.5rem 2rem; border-radius: 14px; color: white; text-align: center; font-size: 1.3rem; font-weight: 600; box-shadow: 0 6px 25px rgba(235,51,73,0.3); }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%); }
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label { color: #e0e0ff !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)
=======

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)
>>>>>>> 7cc97198d2d889b7206651cda6252f27f82fbb8f

# ─────────────────────────────────────────────
# Custom CSS for Premium UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .main-header h1 { color: white; font-weight: 700; font-size: 2rem; margin-bottom: 0.3rem; }
    .main-header p { color: #b8b8d4; font-size: 0.95rem; margin: 0; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Result cards */
    .result-safe {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 6px 25px rgba(0,176,155,0.3);
        animation: pulse-safe 2s infinite;
    }
    .result-danger {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        box-shadow: 0 6px 25px rgba(235,51,73,0.3);
        animation: pulse-danger 1s infinite;
    }
    @keyframes pulse-safe {
        0%, 100% { box-shadow: 0 6px 25px rgba(0,176,155,0.3); }
        50% { box-shadow: 0 6px 35px rgba(0,176,155,0.5); }
    }
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 6px 25px rgba(235,51,73,0.3); }
        50% { box-shadow: 0 6px 35px rgba(235,51,73,0.6); }
    }

    /* Info box */
    .info-box {
        background: linear-gradient(145deg, #1e1e3f, #2d2d5e);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        color: #ccc;
        font-size: 0.85rem;
        margin-top: 1rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #e0e0ff !important;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Gauge section */
    .gauge-section {
        background: rgba(26, 26, 46, 0.6);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
<<<<<<< HEAD
    for p in ["best_model.joblib", "model_building/best_model.joblib"]:
        if os.path.exists(p): return joblib.load(p)
    return None

@st.cache_data
def load_json(filename):
    for p in [filename, f"model_building/{filename}"]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return None

model = load_model()
feature_info = load_json("feature_info.json")
model_comparison = load_json("model_comparison.json")

st.markdown("""
<div class="main-header">
    <h1>🔧 Predictive Engine Maintenance Dashboard</h1>
    <p>AI-powered engine health classification with real-time sensor analysis and batch predictions</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📡 Sensor Inputs")
    st.markdown("---")
    rpm = st.slider("🔄 Engine RPM", 0, 5000, 750, step=10)
    lub_oil_pressure = st.slider("🛢️ Oil Pressure (bar)", 0.0, 10.0, 3.5, step=0.1)
    fuel_pressure = st.slider("⛽ Fuel Pressure (bar)", 0.0, 25.0, 6.0, step=0.1)
    coolant_pressure = st.slider("💧 Coolant Pressure (bar)", 0.0, 10.0, 2.0, step=0.1)
    lub_oil_temp = st.slider("🌡️ Oil Temp (C)", 50.0, 120.0, 78.0, step=0.5)
    coolant_temp = st.slider("❄️ Coolant Temp (C)", 50.0, 120.0, 78.0, step=0.5)
    st.markdown("---")
    if feature_info:
        st.markdown(f"**Model:** `{feature_info.get('best_model_name', 'N/A')}`")
        st.markdown(f"**F1:** `{feature_info.get('best_f1_score', 0):.4f}`")
=======
    paths = ["best_model.joblib", "model_building/best_model.joblib"]
    for p in paths:
        if os.path.exists(p):
            return joblib.load(p)
    return None

@st.cache_data
def load_feature_info():
    paths = ["feature_info.json", "model_building/feature_info.json"]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None

model = load_model()
feature_info = load_feature_info()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔧 Predictive Engine Maintenance Dashboard</h1>
    <p>Real-time engine health classification powered by Machine Learning — enter sensor readings to predict maintenance needs</p>
</div>
""", unsafe_allow_html=True)
>>>>>>> 7cc97198d2d889b7206651cda6252f27f82fbb8f

# ─────────────────────────────────────────────
# Sidebar — Sensor Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Engine Sensor Inputs")
    st.markdown("---")

    rpm = st.slider("🔄 Engine RPM", 0, 5000, 750, step=10, help="Revolutions per minute of the engine")
    lub_oil_pressure = st.slider("🛢️ Lub Oil Pressure (bar)", 0.0, 10.0, 3.5, step=0.1)
    fuel_pressure = st.slider("⛽ Fuel Pressure (bar)", 0.0, 25.0, 6.0, step=0.1)
    coolant_pressure = st.slider("💧 Coolant Pressure (bar)", 0.0, 10.0, 2.0, step=0.1)
    lub_oil_temp = st.slider("🌡️ Lub Oil Temperature (°C)", 50.0, 120.0, 78.0, step=0.5)
    coolant_temp = st.slider("❄️ Coolant Temperature (°C)", 50.0, 120.0, 78.0, step=0.5)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>ℹ️ How it works</strong><br>
        The model uses 6 raw sensor readings plus 3 engineered features
        (Temp/Pressure Ratio, Coolant Efficiency, High RPM Flag) to classify
        engine health in real time.
    </div>
    """, unsafe_allow_html=True)

    if feature_info:
        st.markdown("---")
        st.markdown(f"**🏆 Model:** `{feature_info.get('best_model_name', 'N/A')}`")
        st.markdown(f"**📊 F1 Score:** `{feature_info.get('best_f1_score', 0):.4f}`")

# ─────────────────────────────────────────────
# Build Input DataFrame
# ─────────────────────────────────────────────
input_data = pd.DataFrame([{
    'Engine_RPM': rpm, 'Lub_Oil_Pressure': lub_oil_pressure,
    'Fuel_Pressure': fuel_pressure, 'Coolant_Pressure': coolant_pressure,
    'Lub_Oil_Temperature': lub_oil_temp, 'Coolant_Temperature': coolant_temp
}])
input_data['Temp_Pressure_Ratio'] = input_data['Lub_Oil_Temperature'] / input_data['Lub_Oil_Pressure'].replace(0, np.nan)
input_data['Temp_Pressure_Ratio'] = input_data['Temp_Pressure_Ratio'].fillna(0)
input_data['Coolant_Efficiency'] = input_data['Coolant_Pressure'] / input_data['Coolant_Temperature'].replace(0, np.nan)
input_data['Coolant_Efficiency'] = input_data['Coolant_Efficiency'].fillna(0)
input_data['High_RPM_Flag'] = (input_data['Engine_RPM'] > 1062).astype(int)

<<<<<<< HEAD
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Sensor Gauges", "🏆 Model Comparison", "📁 Batch Predict"])

with tab1:
    col_pred, col_feat = st.columns([1, 1])
    with col_pred:
        st.markdown("### 🎯 Engine Health Prediction")
        if model is not None:
            prediction = model.predict(input_data)
            try:
                proba = model.predict_proba(input_data)[0]
                conf_faulty = proba[1] * 100
            except:
                conf_faulty = 100.0 if prediction[0] == 1 else 0.0
            if prediction[0] == 1:
                st.markdown('<div class="result-danger">🚨 HIGH RISK — Maintenance Required!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-safe">✅ Engine Operating Normally</div>', unsafe_allow_html=True)
            st.markdown("")
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number", value=conf_faulty,
                title={'text': "Failure Risk %", 'font': {'size': 16}},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#eb3349" if conf_faulty > 50 else "#00b09b"},
                       'steps': [{'range': [0, 30], 'color': '#1a3a1a'}, {'range': [30, 70], 'color': '#3a3a1a'}, {'range': [70, 100], 'color': '#3a1a1a'}],
                       'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': 50}}
            ))
            fig_conf.update_layout(height=250, margin=dict(t=40, b=10, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#ccc'})
            st.plotly_chart(fig_conf, use_container_width=True)
            if prediction[0] == 1:
                st.markdown("**Actions:** Schedule inspection, check oil pressure and temperature, review coolant system")
            else:
                st.markdown("**Status:** All parameters normal. Continue regular maintenance schedule.")
        else:
            st.warning("Model not loaded.")
    with col_feat:
        st.markdown("### 🧮 Engineered Features")
        st.dataframe(pd.DataFrame({
            'Feature': ['Temp/Pressure Ratio', 'Coolant Efficiency', 'High RPM Flag'],
            'Value': [f"{input_data['Temp_Pressure_Ratio'].values[0]:.3f}", f"{input_data['Coolant_Efficiency'].values[0]:.4f}", 'Yes' if input_data['High_RPM_Flag'].values[0] == 1 else 'No'],
            'Formula': ['Oil Temp / Oil Pressure', 'Coolant Press / Coolant Temp', 'RPM > 1062']
        }), use_container_width=True, hide_index=True)
        st.markdown("### 📋 Input Vector")
        st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

with tab2:
    st.markdown("### 📊 Real-time Sensor Gauges")
    sensors = [("Engine RPM", rpm, 0, 5000, "RPM"), ("Oil Pressure", lub_oil_pressure, 0, 10, "bar"),
               ("Fuel Pressure", fuel_pressure, 0, 25, "bar"), ("Coolant Pressure", coolant_pressure, 0, 10, "bar"),
               ("Oil Temp", lub_oil_temp, 50, 120, "C"), ("Coolant Temp", coolant_temp, 50, 120, "C")]
    cols = st.columns(3)
    for i, (name, val, lo, hi, unit) in enumerate(sensors):
        with cols[i % 3]:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=val,
                title={'text': name, 'font': {'size': 14}}, number={'suffix': f" {unit}"},
                gauge={'axis': {'range': [lo, hi]}, 'bar': {'color': '#667eea'},
                       'steps': [{'range': [lo, lo+(hi-lo)*0.6], 'color': '#1a2a1a'}, {'range': [lo+(hi-lo)*0.6, lo+(hi-lo)*0.85], 'color': '#2a2a1a'}, {'range': [lo+(hi-lo)*0.85, hi], 'color': '#2a1a1a'}]}
            ))
            fig.update_layout(height=220, margin=dict(t=40, b=5, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': '#ccc'})
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("### 🕸️ Sensor Profile (Normalized)")
    norm_vals = [rpm/5000, lub_oil_pressure/10, fuel_pressure/25, coolant_pressure/10, (lub_oil_temp-50)/70, (coolant_temp-50)/70]
    cats = ['RPM', 'Oil Press', 'Fuel Press', 'Coolant Press', 'Oil Temp', 'Coolant Temp']
    fig_radar = go.Figure(go.Scatterpolar(r=norm_vals+[norm_vals[0]], theta=cats+[cats[0]], fill='toself', fillcolor='rgba(102,126,234,0.3)', line=dict(color='#667eea', width=2)))
    fig_radar.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)', radialaxis=dict(visible=True, range=[0,1], gridcolor='#333')), height=400, margin=dict(t=30,b=30), paper_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'})
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.markdown("### 🏆 Model Performance Comparison")
    if model_comparison and 'results' in model_comparison:
        results = model_comparison['results']
        best = model_comparison.get('best_model', 'Unknown')
        names = list(results.keys())
        f1s = [results[n].get('f1_score', 0) for n in names]
        aucs = [results[n].get('auc_roc', 0) for n in names]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name='F1 Score', x=names, y=f1s, marker_color='#667eea', text=[f'{v:.3f}' for v in f1s], textposition='outside'))
        fig_bar.add_trace(go.Bar(name='AUC-ROC', x=names, y=aucs, marker_color='#764ba2', text=[f'{v:.3f}' for v in aucs], textposition='outside'))
        fig_bar.update_layout(barmode='group', height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'}, yaxis=dict(gridcolor='#333'), legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.success(f"🏆 **Best Model: {best}**")
        st.dataframe(pd.DataFrame({'Model': names, 'F1': [f'{v:.4f}' for v in f1s], 'AUC': [f'{v:.4f}' for v in aucs], 'Best': ['✅' if n==best else '' for n in names]}), use_container_width=True, hide_index=True)
    else:
        st.info("Model comparison data not available. Run training pipeline first.")

with tab4:
    st.markdown("### 📁 Batch Prediction via CSV Upload")
    st.markdown("Upload CSV with: `Engine_RPM`, `Lub_Oil_Pressure`, `Fuel_Pressure`, `Coolant_Pressure`, `Lub_Oil_Temperature`, `Coolant_Temperature`")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded and model is not None:
        batch_df = pd.read_csv(uploaded)
        col_map = {'Engine rpm':'Engine_RPM', 'Lub oil pressure':'Lub_Oil_Pressure', 'Fuel pressure':'Fuel_Pressure', 'Coolant pressure':'Coolant_Pressure', 'lub oil temp':'Lub_Oil_Temperature', 'Coolant temp':'Coolant_Temperature'}
        batch_df.rename(columns=col_map, inplace=True)
        required = ['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']
        if all(c in batch_df.columns for c in required):
            bi = batch_df[required].copy()
            bi['Temp_Pressure_Ratio'] = bi['Lub_Oil_Temperature'] / bi['Lub_Oil_Pressure'].replace(0, np.nan)
            bi['Temp_Pressure_Ratio'] = bi['Temp_Pressure_Ratio'].fillna(0)
            bi['Coolant_Efficiency'] = bi['Coolant_Pressure'] / bi['Coolant_Temperature'].replace(0, np.nan)
            bi['Coolant_Efficiency'] = bi['Coolant_Efficiency'].fillna(0)
            bi['High_RPM_Flag'] = (bi['Engine_RPM'] > 1062).astype(int)
            preds = model.predict(bi)
            try: probas = model.predict_proba(bi)[:,1]
            except: probas = preds.astype(float)
            batch_df['Prediction'] = ['🚨 Faulty' if p==1 else '✅ Normal' for p in preds]
            batch_df['Risk_Score'] = [f'{p*100:.1f}%' for p in probas]
            c1, c2 = st.columns([1,1])
            with c1:
                st.metric("Total", len(preds))
                st.metric("Normal ✅", sum(1 for p in preds if p==0))
                st.metric("Faulty 🚨", sum(1 for p in preds if p==1))
            with c2:
                fig_pie = go.Figure(go.Pie(labels=['Normal','Faulty'], values=[sum(1 for p in preds if p==0), sum(1 for p in preds if p==1)], marker_colors=['#00b09b','#eb3349'], hole=0.5))
                fig_pie.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'}, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Results", batch_df.to_csv(index=False), "predictions.csv", "text/csv")
        else:
            st.error(f"CSV must contain: {required}")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#666;font-size:0.8rem;padding:1rem;"><strong>Predictive Maintenance System</strong> · 7 ML Models · MLflow · SHAP · HuggingFace Spaces<br>© 2026 WildeSoul</div>', unsafe_allow_html=True)
=======
# ─────────────────────────────────────────────
# Sensor Overview Gauges
# ─────────────────────────────────────────────
st.markdown("### 📊 Sensor Overview")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{rpm}</div>
        <div class="metric-label">Engine RPM</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{lub_oil_pressure:.1f}</div>
        <div class="metric-label">Oil Pressure</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{fuel_pressure:.1f}</div>
        <div class="metric-label">Fuel Pressure</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{coolant_pressure:.1f}</div>
        <div class="metric-label">Coolant Press.</div>
    </div>""", unsafe_allow_html=True)
with col5:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{lub_oil_temp:.0f}°</div>
        <div class="metric-label">Oil Temp</div>
    </div>""", unsafe_allow_html=True)
with col6:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{coolant_temp:.0f}°</div>
        <div class="metric-label">Coolant Temp</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
col_pred, col_details = st.columns([1, 1])

with col_pred:
    st.markdown("### 🎯 Prediction Result")
    if model is not None:
        prediction = model.predict(input_data)
        try:
            proba = model.predict_proba(input_data)[0]
            confidence = max(proba) * 100
        except:
            confidence = None

        if prediction[0] == 1:
            st.markdown("""
            <div class="result-danger">
                🚨 HIGH RISK — Engine Maintenance Required!
            </div>
            """, unsafe_allow_html=True)
            if confidence:
                st.markdown(f"**Confidence:** `{confidence:.1f}%`")
            st.markdown("""
            **Recommended Actions:**
            - 🛑 Schedule immediate maintenance inspection
            - 🔍 Check oil pressure and temperature sensors
            - 📋 Review engine coolant system integrity
            """)
        else:
            st.markdown("""
            <div class="result-safe">
                ✅ Engine Operating Normally — No Maintenance Needed
            </div>
            """, unsafe_allow_html=True)
            if confidence:
                st.markdown(f"**Confidence:** `{confidence:.1f}%`")
            st.markdown("""
            **Status Summary:**
            - 🟢 All sensor readings within normal parameters
            - 📅 Continue with regular maintenance schedule
            - 📊 Next recommended check: standard interval
            """)
    else:
        st.warning("⚠️ Model not found. Please run the training pipeline first.")

with col_details:
    st.markdown("### 🧮 Engineered Features")
    eng_df = pd.DataFrame({
        'Feature': ['Temp/Pressure Ratio', 'Coolant Efficiency', 'High RPM Flag'],
        'Value': [
            f"{input_data['Temp_Pressure_Ratio'].values[0]:.3f}",
            f"{input_data['Coolant_Efficiency'].values[0]:.4f}",
            f"{'Yes ⚡' if input_data['High_RPM_Flag'].values[0] == 1 else 'No ✅'}"
        ],
        'Description': [
            'Lub Oil Temp ÷ Lub Oil Pressure',
            'Coolant Pressure ÷ Coolant Temp',
            'RPM > 85th percentile (1062)'
        ]
    })
    st.dataframe(eng_df, use_container_width=True, hide_index=True)

    st.markdown("### 📋 Raw Input Data")
    st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem; padding: 1rem;">
    <strong>Predictive Maintenance System</strong> · Built with Scikit-Learn, XGBoost, LightGBM · 
    Tracked via MLflow · Deployed on HuggingFace Spaces<br>
    © 2026 WildeSoul — AIML Capstone Project
</div>
""", unsafe_allow_html=True)
>>>>>>> 7cc97198d2d889b7206651cda6252f27f82fbb8f
