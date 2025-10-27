import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Parking Dashboard", layout="centered")

@st.cache_resource
def load_vacancy_model():
    return joblib.load('xgb_parking_vacancy_model.pkl')

@st.cache_resource
def load_vehicle_type_model():
    return joblib.load('xgb_vehicle_type_model.pkl')

vacancy_model = load_vacancy_model()
vehicle_type_model = load_vehicle_type_model()



st.title("Smart Parking Occupancy & Vehicle Type Predictor")

st.markdown("""
Welcome to the Smart Parking Prediction Dashboard!

- Predict if a parking slot is **Vacant** or **Occupied** based on date/time features.
- Predict the type of vehicle parked (**Two Wheeler** or **Four Wheeler**).
- Models are trained on augmented datasets for better vacancy prediction.

Adjust the parameters below and click 'Predict' to see the results.
""")

entry_hour = st.slider("Entry Hour (0–23)", 0, 23, 12)
duration = st.number_input("Parking Duration (minutes)", min_value=1, max_value=1440, value=60, step=1)
day_of_week = st.select_slider("Day of Week (0=Monday)", options=list(range(7)), value=0)
is_weekend = 1 if day_of_week >= 5 else 0

bins = [0, 6, 9, 12, 17, 20, 24]
hour_bin = pd.cut([entry_hour], bins=bins, labels=False, right=False)[0]

# Features for Vacancy Model (exclude Duration)
vacancy_features = pd.DataFrame({
    "Entry_Hour": [entry_hour],
    "DayOfWeek": [day_of_week],
    "Is_Weekend": [is_weekend],
    "Hour_Bin": [hour_bin],
})

# Features for Vehicle Type Model (include Duration)
vehicle_features = pd.DataFrame({
    "Entry_Hour": [entry_hour],
    "Duration": [duration],
    "DayOfWeek": [day_of_week],
    "Is_Weekend": [is_weekend],
    "Hour_Bin": [hour_bin],
})

st.subheader("Input Features Preview")
st.write("Vacancy Model Features:")
st.dataframe(vacancy_features)
st.write("Vehicle Type Model Features:")
st.dataframe(vehicle_features)

if st.button("Predict"):
    vacancy_pred = vacancy_model.predict(vacancy_features)[0]
    vehicle_pred = vehicle_type_model.predict(vehicle_features)[0]

    vacancy_status = "Vacant" if vacancy_pred == 1 else "Occupied"
    vehicle_type = "Two Wheeler" if vehicle_pred == 1 else "Four Wheeler"

    col1, col2 = st.columns(2)
    col1.success(f"Parking Slot Status: **{vacancy_status}**")
    col2.success(f"Predicted Vehicle Type: **{vehicle_type}**")

st.markdown("---")
st.header("Project Insights")
st.markdown("""
- **Vacancy Detection Model** uses temporal features, engineered day and hour bins, and was trained on a balanced dataset with forged vacant examples.
- **Vehicle Type Model** predicts between two-wheeler and four-wheeler based on similar features plus parking duration.
- This app helps parking operators optimize utilization and provide real-time information to users.
- The project showcases end-to-end ML lifecycle: data processing, feature engineering, model training, and deployment in Streamlit.

Developed with Python, XGBoost, and Streamlit.
""")
