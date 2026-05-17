import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, json
import plotly.graph_objects as go

st.set_page_config(page_title="Engine Predictive Maintenance", page_icon="🔧", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main-header{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);padding:2rem 2.5rem;border-radius:16px;margin-bottom:1.5rem;color:#fff;box-shadow:0 8px 32px rgba(0,0,0,.25)}
.main-header h1{color:#fff;font-weight:700;font-size:2rem;margin-bottom:.3rem}
.main-header p{color:#b8b8d4;font-size:.95rem;margin:0}
.metric-card{background:linear-gradient(145deg,#1a1a2e,#16213e);border-radius:14px;padding:1.2rem 1.5rem;text-align:center;color:#fff;box-shadow:0 4px 20px rgba(0,0,0,.15);border:1px solid rgba(255,255,255,.05)}
.metric-card .mv{font-size:1.8rem;font-weight:700;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.metric-card .ml{font-size:.78rem;color:#8888aa;text-transform:uppercase;letter-spacing:1px;margin-top:.3rem}
.result-safe{background:linear-gradient(135deg,#00b09b,#96c93d);padding:1.5rem 2rem;border-radius:14px;color:#fff;text-align:center;font-size:1.3rem;font-weight:600}
.result-danger{background:linear-gradient(135deg,#eb3349,#f45c43);padding:1.5rem 2rem;border-radius:14px;color:#fff;text-align:center;font-size:1.3rem;font-weight:600}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0c29,#1a1a3e)}
section[data-testid="stSidebar"] label{color:#e0e0ff!important}
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    for p in ["best_model.joblib","model_building/best_model.joblib"]:
        if os.path.exists(p): return joblib.load(p)
    return None

@st.cache_data
def load_json(fn):
    for p in [fn, f"model_building/{fn}"]:
        if os.path.exists(p):
            with open(p) as f: return json.load(f)
    return None

model = load_model()
feature_info = load_json("feature_info.json")
model_comparison = load_json("model_comparison.json")

st.markdown('<div class="main-header"><h1>🔧 Predictive Engine Maintenance Dashboard</h1><p>AI-powered engine health classification with real-time sensor analysis and batch predictions</p></div>', unsafe_allow_html=True)

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
        st.markdown(f"**Model:** `{feature_info.get('best_model_name','N/A')}`")
        st.markdown(f"**F1:** `{feature_info.get('best_f1_score',0):.4f}`")

inp = pd.DataFrame([{'Engine_RPM':rpm,'Lub_Oil_Pressure':lub_oil_pressure,'Fuel_Pressure':fuel_pressure,'Coolant_Pressure':coolant_pressure,'Lub_Oil_Temperature':lub_oil_temp,'Coolant_Temperature':coolant_temp}])
inp['Temp_Pressure_Ratio'] = inp['Lub_Oil_Temperature'] / inp['Lub_Oil_Pressure'].replace(0, np.nan)
inp['Temp_Pressure_Ratio'] = inp['Temp_Pressure_Ratio'].fillna(0)
inp['Coolant_Efficiency'] = inp['Coolant_Pressure'] / inp['Coolant_Temperature'].replace(0, np.nan)
inp['Coolant_Efficiency'] = inp['Coolant_Efficiency'].fillna(0)
inp['High_RPM_Flag'] = (inp['Engine_RPM'] > 1062).astype(int)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction","📊 Sensor Gauges","🏆 Model Comparison","📁 Batch Predict"])

with tab1:
    c1, c2 = st.columns([1,1])
    with c1:
        st.markdown("### 🎯 Engine Health Prediction")
        if model is not None:
            pred = model.predict(inp)
            try:
                proba = model.predict_proba(inp)[0]; cf = proba[1]*100
            except:
                cf = 100.0 if pred[0]==1 else 0.0
            if pred[0]==1:
                st.markdown('<div class="result-danger">🚨 HIGH RISK - Maintenance Required!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-safe">✅ Engine Operating Normally</div>', unsafe_allow_html=True)
            st.markdown("")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=cf, title={'text':"Failure Risk %",'font':{'size':16}},
                gauge={'axis':{'range':[0,100]},'bar':{'color':"#eb3349" if cf>50 else "#00b09b"},
                'steps':[{'range':[0,30],'color':'#1a3a1a'},{'range':[30,70],'color':'#3a3a1a'},{'range':[70,100],'color':'#3a1a1a'}],
                'threshold':{'line':{'color':"white",'width':3},'thickness':0.8,'value':50}}))
            fig.update_layout(height=250, margin=dict(t=40,b=10,l=30,r=30), paper_bgcolor='rgba(0,0,0,0)', font={'color':'#ccc'})
            st.plotly_chart(fig, use_container_width=True)
            if pred[0]==1: st.markdown("**Actions:** Schedule inspection, check oil pressure and temperature")
            else: st.markdown("**Status:** All parameters normal. Continue regular maintenance.")
        else: st.warning("Model not loaded.")
    with c2:
        st.markdown("### 🧮 Engineered Features")
        st.dataframe(pd.DataFrame({'Feature':['Temp/Pressure Ratio','Coolant Efficiency','High RPM Flag'],'Value':[f"{inp['Temp_Pressure_Ratio'].values[0]:.3f}",f"{inp['Coolant_Efficiency'].values[0]:.4f}",'Yes' if inp['High_RPM_Flag'].values[0]==1 else 'No'],'Formula':['Oil Temp / Oil Press','Coolant Press / Coolant Temp','RPM > 1062']}), use_container_width=True, hide_index=True)
        st.markdown("### 📋 Input Vector")
        st.dataframe(inp.T.rename(columns={0:'Value'}), use_container_width=True)

with tab2:
    st.markdown("### 📊 Real-time Sensor Gauges")
    sensors = [("Engine RPM",rpm,0,5000,"RPM"),("Oil Pressure",lub_oil_pressure,0,10,"bar"),("Fuel Pressure",fuel_pressure,0,25,"bar"),("Coolant Pressure",coolant_pressure,0,10,"bar"),("Oil Temp",lub_oil_temp,50,120,"C"),("Coolant Temp",coolant_temp,50,120,"C")]
    cols = st.columns(3)
    for i,(nm,vl,lo,hi,un) in enumerate(sensors):
        with cols[i%3]:
            fg = go.Figure(go.Indicator(mode="gauge+number",value=vl,title={'text':nm,'font':{'size':14}},number={'suffix':f" {un}"},gauge={'axis':{'range':[lo,hi]},'bar':{'color':'#667eea'},'steps':[{'range':[lo,lo+(hi-lo)*0.6],'color':'#1a2a1a'},{'range':[lo+(hi-lo)*0.6,lo+(hi-lo)*0.85],'color':'#2a2a1a'},{'range':[lo+(hi-lo)*0.85,hi],'color':'#2a1a1a'}]}))
            fg.update_layout(height=220,margin=dict(t=40,b=5,l=20,r=20),paper_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'})
            st.plotly_chart(fg, use_container_width=True)
    st.markdown("### 🕸️ Sensor Profile")
    nv = [rpm/5000,lub_oil_pressure/10,fuel_pressure/25,coolant_pressure/10,(lub_oil_temp-50)/70,(coolant_temp-50)/70]
    ct = ['RPM','Oil Press','Fuel Press','Coolant Press','Oil Temp','Coolant Temp']
    fr = go.Figure(go.Scatterpolar(r=nv+[nv[0]],theta=ct+[ct[0]],fill='toself',fillcolor='rgba(102,126,234,0.3)',line=dict(color='#667eea',width=2)))
    fr.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',radialaxis=dict(visible=True,range=[0,1],gridcolor='#333')),height=400,margin=dict(t=30,b=30),paper_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'})
    st.plotly_chart(fr, use_container_width=True)

with tab3:
    st.markdown("### 🏆 Model Performance Comparison")
    if model_comparison and 'results' in model_comparison:
        r = model_comparison['results']; b = model_comparison.get('best_model','')
        ns = list(r.keys()); f1s = [r[n].get('f1_score',0) for n in ns]; acs = [r[n].get('auc_roc',0) for n in ns]
        fb = go.Figure()
        fb.add_trace(go.Bar(name='F1',x=ns,y=f1s,marker_color='#667eea',text=[f'{v:.3f}' for v in f1s],textposition='outside'))
        fb.add_trace(go.Bar(name='AUC',x=ns,y=acs,marker_color='#764ba2',text=[f'{v:.3f}' for v in acs],textposition='outside'))
        fb.update_layout(barmode='group',height=400,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'},yaxis=dict(gridcolor='#333'))
        st.plotly_chart(fb, use_container_width=True)
        st.success(f"🏆 **Best Model: {b}**")
        st.dataframe(pd.DataFrame({'Model':ns,'F1':[f'{v:.4f}' for v in f1s],'AUC':[f'{v:.4f}' for v in acs],'Best':['✅' if n==b else '' for n in ns]}),use_container_width=True,hide_index=True)
    else: st.info("Run training pipeline to generate model comparison data.")

with tab4:
    st.markdown("### 📁 Batch Prediction via CSV")
    st.markdown("Upload CSV with: Engine_RPM, Lub_Oil_Pressure, Fuel_Pressure, Coolant_Pressure, Lub_Oil_Temperature, Coolant_Temperature")
    up = st.file_uploader("Upload CSV", type=['csv'])
    if up and model:
        df = pd.read_csv(up)
        df.rename(columns={'Engine rpm':'Engine_RPM','Lub oil pressure':'Lub_Oil_Pressure','Fuel pressure':'Fuel_Pressure','Coolant pressure':'Coolant_Pressure','lub oil temp':'Lub_Oil_Temperature','Coolant temp':'Coolant_Temperature'}, inplace=True)
        req = ['Engine_RPM','Lub_Oil_Pressure','Fuel_Pressure','Coolant_Pressure','Lub_Oil_Temperature','Coolant_Temperature']
        if all(c in df.columns for c in req):
            bi = df[req].copy()
            bi['Temp_Pressure_Ratio'] = bi['Lub_Oil_Temperature']/bi['Lub_Oil_Pressure'].replace(0,np.nan); bi['Temp_Pressure_Ratio']=bi['Temp_Pressure_Ratio'].fillna(0)
            bi['Coolant_Efficiency'] = bi['Coolant_Pressure']/bi['Coolant_Temperature'].replace(0,np.nan); bi['Coolant_Efficiency']=bi['Coolant_Efficiency'].fillna(0)
            bi['High_RPM_Flag'] = (bi['Engine_RPM']>1062).astype(int)
            ps = model.predict(bi)
            try: pb = model.predict_proba(bi)[:,1]
            except: pb = ps.astype(float)
            df['Prediction'] = ['🚨 Faulty' if p==1 else '✅ Normal' for p in ps]
            df['Risk'] = [f'{p*100:.1f}%' for p in pb]
            ca,cb = st.columns(2)
            with ca: st.metric("Total",len(ps)); st.metric("Normal",sum(1 for p in ps if p==0)); st.metric("Faulty",sum(1 for p in ps if p==1))
            with cb:
                fp = go.Figure(go.Pie(labels=['Normal','Faulty'],values=[sum(1 for p in ps if p==0),sum(1 for p in ps if p==1)],marker_colors=['#00b09b','#eb3349'],hole=0.5))
                fp.update_layout(height=300,paper_bgcolor='rgba(0,0,0,0)',font={'color':'#ccc'},showlegend=False)
                st.plotly_chart(fp, use_container_width=True)
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button("📥 Download Results",df.to_csv(index=False),"predictions.csv","text/csv")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#666;font-size:.8rem;padding:1rem"><b>Predictive Maintenance</b> | 7 ML Models | MLflow | SHAP | HuggingFace<br>2026 WildeSoul</div>',unsafe_allow_html=True)
