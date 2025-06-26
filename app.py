import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURASI HALAMAN (Tetap sama) ---
st.set_page_config(
    page_title="Prediksi Kanker Payudara",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI UNTUK MEMUAT ASET (Tetap sama) ---
@st.cache_resource
def load_assets():
    """Fungsi ini memuat model dan scaler yang telah disimpan."""
    try:
        # Ganti nama file ini jika Anda menggunakan model stacking
        model = joblib.load('stacking_classifer.pkl') 
        scaler = joblib.load('scaler_wisconsin.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: File model atau scaler tidak ditemukan.")
        return None, None

model, scaler = load_assets()


# --- FUNGSI INPUT PENGGUNA (INI YANG DIPERBAIKI) ---
def user_input_features():
    """Membuat widget input di sidebar untuk semua 30 fitur."""
    
    # Label di sidebar tetap deskriptif untuk pengguna
    st.sidebar.header('Input Fitur Sel Tumor')
    
    st.sidebar.subheader("Fitur Rata-rata (Mean)")
    radius_mean = st.sidebar.slider('Radius (Mean)', 6.9, 28.2, 17.0)
    texture_mean = st.sidebar.slider('Texture (Mean)', 9.7, 39.3, 19.5)
    # ... (lanjutkan semua slider seperti sebelumnya) ...
    # ...
    # ... (Saya akan singkat untuk contoh, tapi Anda harus memasukkan semua 30 slider)
    perimeter_mean = st.sidebar.slider('Perimeter (Mean)', 43.7, 188.6, 115.0)
    area_mean = st.sidebar.slider('Area (Mean)', 143.0, 2501.0, 950.0)
    smoothness_mean = st.sidebar.slider('Smoothness (Mean)', 0.05, 0.17, 0.1)
    compactness_mean = st.sidebar.slider('Compactness (Mean)', 0.01, 0.35, 0.15)
    concavity_mean = st.sidebar.slider('Concavity (Mean)', 0.0, 0.43, 0.1)
    concave_points_mean = st.sidebar.slider('Concave Points (Mean)', 0.0, 0.21, 0.05)
    symmetry_mean = st.sidebar.slider('Symmetry (Mean)', 0.1, 0.31, 0.18)
    fractal_dimension_mean = st.sidebar.slider('Fractal Dimension (Mean)', 0.04, 0.1, 0.06)

    st.sidebar.subheader("Fitur Standar Error (SE)")
    radius_se = st.sidebar.slider('Radius (SE)', 0.11, 2.88, 0.5)
    texture_se = st.sidebar.slider('Texture (SE)', 0.36, 4.89, 1.2)
    perimeter_se = st.sidebar.slider('Perimeter (SE)', 0.75, 21.99, 3.5)
    area_se = st.sidebar.slider('Area (SE)', 6.8, 542.5, 50.0)
    smoothness_se = st.sidebar.slider('Smoothness (SE)', 0.001, 0.032, 0.007)
    compactness_se = st.sidebar.slider('Compactness (SE)', 0.002, 0.136, 0.025)
    concavity_se = st.sidebar.slider('Concavity (SE)', 0.0, 0.4, 0.03)
    concave_points_se = st.sidebar.slider('Concave Points (SE)', 0.0, 0.053, 0.012)
    symmetry_se = st.sidebar.slider('Symmetry (SE)', 0.007, 0.079, 0.02)
    fractal_dimension_se = st.sidebar.slider('Fractal Dimension (SE)', 0.0008, 0.03, 0.003)

    st.sidebar.subheader("Fitur Terburuk (Worst)")
    radius_worst = st.sidebar.slider('Radius (Worst)', 7.9, 36.1, 22.0)
    texture_worst = st.sidebar.slider('Texture (Worst)', 12.0, 49.6, 25.5)
    perimeter_worst = st.sidebar.slider('Perimeter (Worst)', 50.4, 251.5, 150.0)
    area_worst = st.sidebar.slider('Area (Worst)', 185.0, 4255.0, 1500.0)
    smoothness_worst = st.sidebar.slider('Smoothness (Worst)', 0.07, 0.23, 0.13)
    compactness_worst = st.sidebar.slider('Compactness (Worst)', 0.02, 1.06, 0.25)
    concavity_worst = st.sidebar.slider('Concavity (Worst)', 0.0, 1.26, 0.28)
    concave_points_worst = st.sidebar.slider('Concave Points (Worst)', 0.0, 0.3, 0.15)
    symmetry_worst = st.sidebar.slider('Symmetry (Worst)', 0.15, 0.67, 0.29)
    fractal_dimension_worst = st.sidebar.slider('Fractal Dimension (Worst)', 0.05, 0.21, 0.08)


    # === PERUBAHAN UTAMA ADA DI SINI ===
    # Buat dictionary dengan NAMA KOLOM YANG BENAR (sesuai yang diharapkan model)
    data = {
        'radius1': radius_mean, 'texture1': texture_mean, 'perimeter1': perimeter_mean, 'area1': area_mean, 'smoothness1': smoothness_mean,
        'compactness1': compactness_mean, 'concavity1': concavity_mean, 'concave_points1': concave_points_mean, 'symmetry1': symmetry_mean, 'fractal_dimension1': fractal_dimension_mean,
        'radius2': radius_se, 'texture2': texture_se, 'perimeter2': perimeter_se, 'area2': area_se, 'smoothness2': smoothness_se,
        'compactness2': compactness_se, 'concavity2': concavity_se, 'concave_points2': concave_points_se, 'symmetry2': symmetry_se, 'fractal_dimension2': fractal_dimension_se,
        'radius3': radius_worst, 'texture3': texture_worst, 'perimeter3': perimeter_worst, 'area3': area_worst, 'smoothness3': smoothness_worst,
        'compactness3': compactness_worst, 'concavity3': concavity_worst, 'concave_points3': concave_points_worst, 'symmetry3': symmetry_worst, 'fractal_dimension3': fractal_dimension_worst
    }
    
    # Mengubah dictionary menjadi DataFrame dengan nama kolom yang sudah benar
    features = pd.DataFrame(data, index=[0])
    return features

# --- HALAMAN UTAMA (Tetap sama, tidak perlu diubah) ---
st.title("🔬 Prediksi Kanker Payudara")
# ... (sisa kode halaman utama Anda tetap sama) ...

if model is not None and scaler is not None:
    input_df = user_input_features()
    st.subheader('Parameter Input yang Anda Masukkan:')
    st.write(input_df)
    if st.button('Lakukan Prediksi'):
        # Normalisasi sekarang akan berhasil karena nama kolom sudah cocok
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        
        st.subheader('Hasil Prediksi Diagnosis:')
        if prediction[0] == 1:
            st.error("Hasil: Ganas (Malignant)")
        else:
            st.success("Hasil: Jinak (Benign)")