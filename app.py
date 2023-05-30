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
    
    insured_zip = st.text_input("Insured ZIP")
    collision_type = st.selectbox("Collision Type", ["Rear Collision", "Front Collision", "Side Collision"])
    incident_severity = st.selectbox("Incident Severity", ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"])
    authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"])
    insured_sex = st.selectbox("Insured Sex", ["Male", "Female"])
    capital_gains = st.number_input("Capital Gains", step=1.0, format="%.2f")
    capital_loss = st.number_input("Capital Loss", step=1.0, format="%.2f")
    insured_relationship = st.selectbox("Insured Relationship", ["Spouse", "Own Child", "Not-in-family", "Unmarried", "Other-relative"])
    policy_state = st.selectbox("Policy State", ["IL", "IN", "OH"])
    umbrella_limit = st.number_input("Umbrella Limit", step=1, format="%d")
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft"])
    insured_education_level = st.selectbox("Insured Education Level", ["High School", "College", "Associate", "MD", "PhD", "Masters"])
    policy_deductable = st.number_input("Policy Deductible", step=1, format="%d")
    policy_annual_premium = st.number_input("Policy Annual Premium", step=0.01, format="%.2f")

    if st.button("Predict"):
        data = CustomData(
            insured_zip=insured_zip,
            collision_type=collision_type,
            incident_severity=incident_severity,
            authorities_contacted=authorities_contacted,
            insured_sex=insured_sex,
            capital_gains=float(capital_gains),
            capital_loss=float(capital_loss),
            insured_relationship=insured_relationship,
            policy_state=policy_state,
            umbrella_limit=int(umbrella_limit),
            incident_type=incident_type,
            insured_education_level=insured_education_level,
            policy_deductable=int(policy_deductable),
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
