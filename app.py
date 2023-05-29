import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def main():
    st.title("Insurance Prediction Web App")

    st.write("Enter the details to predict insurance results:")

    sex = st.selectbox("Sex", ["Male", "Female"])
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft"])
    collision_type = st.selectbox("Collision Type", ["Rear Collision", "Side Collision", "Front Collision"])
    incident_severity = st.selectbox("Incident Severity", ["Minor Damage", "Major Damage", "Total Loss"])
    incident_city = st.text_input("Incident City")
    num_vehicles_involved = st.number_input("Number of Vehicles Involved", min_value=1, step=1)
    property_damage = st.number_input("Property Damage", min_value=0.0, step=0.01)
    police_report_available = st.number_input("Police Report Available", min_value=0.0, step=0.01)
    bodily_injuries = st.number_input("Bodily Injuries", min_value=0.0, step=0.01)

    if st.button("Predict"):
        data = CustomData(
            insured_sex=sex,
            incident_type=incident_type,
            collision_type=collision_type,
            incident_severity=incident_severity,
            incident_city=incident_city,
            number_of_vehicles_involved=num_vehicles_involved,
            property_damage=property_damage,
            police_report_available=police_report_available,
            bodily_injuries=bodily_injuries
        )

        pred_df = data.get_data_as_data_frame()
        st.write(pred_df)
        st.write("Before Prediction")

        predict_pipeline = PredictPipeline()
        st.write("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        st.write("After Prediction")
        st.write("Prediction:", results[0])

if __name__ == "__main__":
    main()
