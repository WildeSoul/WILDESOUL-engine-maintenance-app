import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Engine Maintenance Predictor", layout="wide")

st.title("Predictive Maintenance for Engine Health")
st.write("This app predicts whether an engine requires maintenance based on sensor readings.")

# Load Model
@st.cache_resource
def load_model():
    # Make sure you have the model downloaded from HF space or available locally
    try:
        model = joblib.load("best_model.joblib")
        return model
    except:
        return None

model = load_model()

# Sidebar for inputs
st.sidebar.header("Engine Sensor Inputs")
rpm = st.sidebar.number_input("Engine RPM", 0.0, 10000.0, 2500.0)
lub_oil_pressure = st.sidebar.number_input("Lub Oil Pressure", 0.0, 10.0, 3.5)
fuel_pressure = st.sidebar.number_input("Fuel Pressure", 0.0, 10.0, 4.0)
coolant_pressure = st.sidebar.number_input("Coolant Pressure", 0.0, 10.0, 2.0)
lub_oil_temp = st.sidebar.number_input("Lub Oil Temperature", 0.0, 200.0, 85.0)
coolant_temp = st.sidebar.number_input("Coolant Temperature", 0.0, 200.0, 80.0)

input_data = pd.DataFrame({
    'Engine_RPM': [rpm],
    'Lub_Oil_Pressure': [lub_oil_pressure],
    'Fuel_Pressure': [fuel_pressure],
    'Coolant_Pressure': [coolant_pressure],
    'Lub_Oil_Temperature': [lub_oil_temp],
    'Coolant_Temperature': [coolant_temp]
})

st.subheader("Input Values")
st.write(input_data)

if st.button("Predict"):
    if model is not None:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("🚨 ALERT: Maintenance Required. Engine is Faulty.")
        else:
            st.success("✅ Normal Operation. Engine is Healthy.")
    else:
        st.warning("Model not found. Please ensure 'best_model.joblib' is in the same directory.")
