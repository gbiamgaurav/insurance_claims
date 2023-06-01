import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

## Route for Home page

def index():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")

def predict_datapoint():
    st.title("Predict Data Point")
    st.write("Enter the required information:")
    
    
    incident_severity = st.selectbox("Incident Severity", ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"])
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft"])
    policy_annual_premium = st.number_input("Enter a number")

    if st.button("Predict"):
        data = CustomData(
            
            incident_severity=incident_severity,
            incident_type=incident_type,
            policy_annual_premium=float(policy_annual_premium)
        )

        pred_df = data.get_data_as_dataframe()
        st.write(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.write(results[0])

def main():
    st.sidebar.title("Insurance Prediction App")
    page = st.sidebar.selectbox("Select a page", ["Home", "Predict Data Point"])

    if page == "Home":
        index()
    elif page == "Predict Data Point":
        predict_datapoint()

if __name__ == "__main__":
    main()
