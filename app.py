import streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

def main():
    st.title("Insurance Fraud Detection")
    st.subheader("Enter the details for prediction")

    insured_sex = st.selectbox("Insured Sex", ["Male", "Female"])
    incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision",
                                                   "Parked Car"])
    collision_type = st.selectbox("Collision Type", ["Front Collision", "Rear Collision",
                                                     "Side Collision", "Unknown"])
    incident_severity = st.selectbox("Incident Severity", ["Major Damage", "Minor Damage",
                                                           "Total Loss", "Trivial Damage"])
    incident_city = st.selectbox("Incident City", ["Arlington", "Springfield", "Hillsdale",
                                                   "Northbend", "Riverwood"])
    number_of_vehicles_involved = st.number_input("Number of Vehicles Involved", min_value=1, step=1, max_value=5)
    property_damage = st.selectbox("Property Damage", ["Yes", "No"])
    police_report_available = st.selectbox("Police Report Available", ["Yes", "No"])
    bodily_injuries = st.number_input("Number of Bodily Injuries", min_value=0, step=1, max_value=3)

    custom_data = CustomData(insured_sex, incident_type, collision_type, incident_severity, incident_city,
                             number_of_vehicles_involved, property_damage, police_report_available, bodily_injuries)

    if st.button("Predict"):
        pred_pipeline = PredictPipeline()
        input_data = custom_data.get_data_as_dataframe()
        prediction = pred_pipeline.predict(input_data)
        if prediction[0] == 1:
            st.error("Fraudulent Insurance Claim Detected")
        else:
            st.success("Legitimate Insurance Claim")

if __name__ == "__main__":
    main()
