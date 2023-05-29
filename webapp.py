# import the necessary libraries:
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the preprocessor object and the trained model
preprocessor = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

# Define the web app's header and description
st.title("Fraud Detection Web App")
st.write("Predict whether a claim is fraudulent or not based on selected features.")

# Create input fields for each selected feature and get user inputs
insured_zip = st.number_input("Insured Zip")
collision_type = st.selectbox("Collision Type", ["Front Collision", "Rear Collision", "Side Collision"])
incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Major Damage", "Total Loss"])
authorities_contacted = st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "None"])
insured_sex = st.selectbox("Insured Sex", ["Male", "Female"])
capital_gains = st.number_input("Capital Gains")
capital_loss = st.number_input("Capital Loss")
insured_relationship = st.selectbox("Insured Relationship", ["Self", "Spouse", "Child", "Other"])
policy_state = st.selectbox("Policy State", ["IL", "IN", "OH"])
umbrella_limit = st.selectbox("Umbrella Limit", ["$0", "$1-9999", "$10000", "$20000"])
incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car"])
insured_education_level = st.selectbox("Insured Education Level", ["High School", "College", "Associate", "Bachelor", "Master", "Doctorate"])
policy_deductable = st.number_input("Policy Deductable")
policy_annual_premium = st.number_input("Policy Annual Premium")

# Create a DataFrame with the user inputs and apply the preprocessor
input_data = pd.DataFrame({
    "insured_zip": [insured_zip],
    "collision_type": [collision_type],
    "incident_severity": [incident_severity],
    "authorities_contacted": [authorities_contacted],
    "insured_sex": [insured_sex],
    "capital-gains": [capital_gains],
    "capital-loss": [capital_loss],
    "insured_relationship": [insured_relationship],
    "policy_state": [policy_state],
    "umbrella_limit": [umbrella_limit],
    "incident_type": [incident_type],
    "insured_education_level": [insured_education_level],
    "policy_deductable": [policy_deductable],
    "policy_annual_premium": [policy_annual_premium]
})

input_transformed = preprocessor.transform(input_data)

# Make predictions using the transformed input data
prediction = model.predict(input_transformed)
fraud_probability = model.predict_proba(input_transformed)[:, 1]

# Display the prediction result
if prediction[0] == 1:
    st.write("Fraud Detected!")
    st.write("The probability of fraud: ", fraud_probability[0])
else:
    st.write("No Fraud Detected")


# Run the Streamlit app
if __name__ == "__main__":
    webapp.py




