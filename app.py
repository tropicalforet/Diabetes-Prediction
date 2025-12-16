import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load(open("diabetes_model.pkl", "rb"))

st.title("Prediksi Diabetes - Machine Learning")
st.write("Masukkan data berikut untuk memprediksi risiko diabetes.")

# Input form
gender = st.radio("Jenis Kelamin", ["Male", "Female", "Other"])
age = st.number_input("Umur", min_value=1, max_value=120, value=30)
hypertension = st.radio("Hipertensi", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
heart_disease = st.radio("Penyakit Jantung", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
smoking_history = st.selectbox("Riwayat Merokok", ["never", "No Info", "current", "former", "ever", "not current"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
hba1c_level = st.number_input("HbA1c Level", min_value=3.5, max_value=9.0, value=5.7, format="%.1f")
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=70, max_value=300, value=140)

# Predict button
if st.button("Predict"):    
    # Prepare input data as a dictionary
    input_data = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }

    # Convert dictionary to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    st.subheader("Prediction Results:")
    if prediction[0] == 1:
        st.error(f"Hasil Prediksi: Anda berisiko Diabetes (Probabilitas: {prediction_proba[0]*100:.2f}%)")
    else:
        st.success(f"Hasil Prediksi: Anda tidak berisiko Diabetes (Probabilitas: {(1-prediction_proba[0])*100:.2f}%)")

    st.write("Disclaimer: This prediction is based on a machine learning model and should not be considered as medical advice.")
