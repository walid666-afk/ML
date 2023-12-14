# streamlit_dengue_diagnosis.py
import streamlit as st
import pickle
import numpy as np

# Load model dan normalisasi dengan pickle
with open('logregmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('norm.pkl', 'rb') as normalization_file:
    normalization = pickle.load(normalization_file)

# Fungsi untuk melakukan prediksi
def predict_dengue(features):
    # Lakukan normalisasi terlebih dahulu
    features = normalization.transform([features])

    # Lakukan prediksi
    prediction = model.predict(features)
    return prediction

# Fungsi utama aplikasi Streamlit
def main():
    st.title("Diagnosis Penyakit Diabetes Mellitus")

    # Input fitur dari pengguna
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=100, value=0)
    glucose = st.number_input("Glucose", min_value=0.0, max_value=100.0, value=0.0)
    bloodpressure = st.number_input("Blood Pressure",min_value=0.0, max_value=100.0, value=0.0)
    skin = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=0.0)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=100.0, value=0.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=300.0, value=0.0)
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=0.0)


    # Tampilkan tombol submit
    submit_button = st.button("Diagnosa")

    # Jika tombol submit ditekan, lakukan prediksi
    if submit_button:
        # Prediksi menggunakan model
        features = [pregnancies,glucose,bloodpressure, skin, insulin, bmi,age]
        result = predict_dengue(features)

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        if result == 1:
            st.write("<div style='color: red; font-size: 18px; font-weight: bold;'>Pasien kemungkinan menderita Diabetes Mellitus.</div>", unsafe_allow_html=True)
        else:
            st.write("<div style='color: green; font-size: 18px; font-weight: bold;'>Pasien kemungkinan tidak menderita Diabetes Mellitus.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
