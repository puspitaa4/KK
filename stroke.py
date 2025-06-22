import streamlit as st
import joblib
import numpy as np

# Load model dari file .pkl / .skl
model = joblib.load("naive_bayes_stroke_model.pkl")  # Ganti nama file jika berbeda

# Judul aplikasi
st.title("Deteksi Dini Penyakit Stroke")

# Form input fitur
id = st.number_input("ID Pasien", min_value=0)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
age = st.number_input("Usia (dalam tahun)", min_value=0.0, max_value=120.0, step=0.1)
hypertension = st.selectbox("Riwayat Hipertensi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
heart_disease = st.selectbox("Riwayat Penyakit Jantung", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa", min_value=0.0, step=0.1)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, step=0.1)
smoking_status = st.selectbox("Status Merokok", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Tombol Prediksi
if st.button("Prediksi Stroke"):
    # Preprocessing jika diperlukan (contoh encoding manual)
    gender_map = {"Male": 1, "Female": 0}
    smoking_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}

    # Bentuk array input sesuai urutan fitur saat training
    input_data = np.array([
        id,
        gender_map[gender],
        age,
        hypertension,
        heart_disease,
        avg_glucose_level,
        bmi,
        smoking_map[smoking_status]
    ]).reshape(1, -1)

    # Prediksi
    prediction = model.predict(input_data)

    # Tampilkan hasil
    if prediction[0] == 1:
        st.error("Berpotensi terkena stroke.")
    else:
        st.success("Tidak berpotensi terkena stroke.")
