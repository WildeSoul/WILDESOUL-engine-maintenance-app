import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Engine Predictive Maintenance", layout="wide")
st.title("⚙️ Predictive Engine Maintenance Dashboard")

@st.cache_resource
def load_model():
    if os.path.exists("best_model.joblib"):
        return joblib.load("best_model.joblib")
    elif os.path.exists("model_building/best_model.joblib"):
        return joblib.load("model_building/best_model.joblib")
    return None

model = load_model()

st.sidebar.header("Engine Sensor Inputs")
rpm = st.sidebar.number_input("Engine RPM", 0.0, 10000.0, 750.0)
lub_oil_pressure = st.sidebar.number_input("Lub Oil Pressure", 0.0, 10.0, 3.5)
fuel_pressure = st.sidebar.number_input("Fuel Pressure", 0.0, 10.0, 4.0)
coolant_pressure = st.sidebar.number_input("Coolant Pressure", 0.0, 10.0, 2.0)
lub_oil_temp = st.sidebar.number_input("Lub Oil Temperature", 0.0, 200.0, 85.0)
coolant_temp = st.sidebar.number_input("Coolant Temperature", 0.0, 200.0, 80.0)

input_data = pd.DataFrame([{
    'Engine_RPM': rpm,
    'Lub_Oil_Pressure': lub_oil_pressure,
    'Fuel_Pressure': fuel_pressure,
    'Coolant_Pressure': coolant_pressure,
    'Lub_Oil_Temperature': lub_oil_temp,
    'Coolant_Temperature': coolant_temp
}])

# Feature Engineering (must match training pipeline)
input_data['Temp_Pressure_Ratio'] = input_data['Lub_Oil_Temperature'] / input_data['Lub_Oil_Pressure'].replace(0, np.nan)
input_data['Temp_Pressure_Ratio'] = input_data['Temp_Pressure_Ratio'].fillna(0)
input_data['Coolant_Efficiency'] = input_data['Coolant_Pressure'] / input_data['Coolant_Temperature'].replace(0, np.nan)
input_data['Coolant_Efficiency'] = input_data['Coolant_Efficiency'].fillna(0)
input_data['High_RPM_Flag'] = (input_data['Engine_RPM'] > 1062).astype(int)

if st.button("Predict"):
    if model is not None:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.error("🚨 HIGH RISK: Engine Failure Predicted!")
        else:
            st.success("✅ Engine Operating Normally")
    else:
        st.warning("Model not found. Please run the training pipeline first.")
