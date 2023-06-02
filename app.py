import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

FRAUD_LABEL = "Fraud"
NO_FRAUD_LABEL = "No Fraud"


def index():
    st.title("Home Page")
    st.write("Welcome to the Home Page!")


def predict_datapoint():
    st.title("Insurance Claim Prediction")
    st.write("Enter the required information:")

    insured_sex = st.selectbox("Sex", ["FEMALE", "MALE"])
    insured_relationship = st.selectbox("Relationship", ["own-child", "other-relative", "not-in-family", "husband", "wife", "unmarried"])
    collision_type = st.selectbox("Collision Type", ["Rear Collision", "Side Collision", "Front Collision", "Others"])
    incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Total Loss", "Major Damage", "Trivial Damage"])
    incident_state = st.selectbox("Incident State", ["NY","SC","WV","VA","NC","PA","OH"])
    incident_city = st.selectbox("Incident City", ["Springfield","Arlington","Columbus","Northbend","Hillsdale","Riverwood","Northbrook"])
    property_damage = st.selectbox("Property Damage", ["Others", "NO", "YES"])
    police_report_available = st.selectbox("Police report available", ["Not Available", "NO", "YES"])
    bodily_injuries = st.slider("Bodily Injuries", 0, 2, 1)
    witnesses = st.slider("Witnesses", 0, 3, 1)

    if st.button("Predict"):
        data = CustomData(
            insured_sex=insured_sex,
            insured_relationship=insured_relationship,
            collision_type=collision_type,
            incident_severity=incident_severity,
            incident_state=incident_state,
            incident_city=incident_city,
            property_damage=property_damage,
            police_report_available=police_report_available,
            bodily_injuries=bodily_injuries,
            witnesses=witnesses
        )

        pred_df = data.get_data_as_dataframe()

        # Update the prediction pipeline to assign 1 for "Fraud" and 0 for "No Fraud"
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Convert result to pandas Series and use replace()
        results[0] = pd.Series(results[0]).replace({NO_FRAUD_LABEL: 0, FRAUD_LABEL: 1})

        # Display the predicted results
        st.write("Predicted Result:", results[0])
        
        # Display the predicted results
        if results[0] == 0:
            st.write("Predicted Result: No Fraud")
        else:
            st.write("Predicted Result: Fraud")


def main():
    st.sidebar.title("Insurance Prediction App")
    page = st.sidebar.selectbox("Select a page", ["Home", "Predict Data Point"])

    if page == "Home":
        index()
    elif page == "Predict Data Point":
        predict_datapoint()


if __name__ == "__main__":
    main()
