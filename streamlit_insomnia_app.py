import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Aplikasi Prediksi Risiko Insomnia Mahasiswa")
st.write("Menggunakan Regresi Linier dan Streamlit")

# Upload file CSV
data_file = st.file_uploader("Unggah file dataset (.csv)", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)
    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # Split fitur dan target
    X = df.drop(columns=["insomnia_risk"])
    y = df["insomnia_risk"]

    # Split data latih dan uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat model regresi linier
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Evaluasi Model")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"R-squared (R²): {r2:.4f}")

    # Visualisasi scatter plot prediksi vs aktual
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Nilai Aktual")
    ax.set_ylabel("Nilai Prediksi")
    ax.set_title("Scatter Plot: Aktual vs Prediksi")
    st.pyplot(fig)

    # Koefisien
    st.subheader("Koefisien Model")
    coef_df = pd.DataFrame({"Fitur": X.columns, "Koefisien": model.coef_})
    st.dataframe(coef_df)

    # Input manual
    st.subheader("Input Data Manual untuk Prediksi")
    input_data = {}
    for col in X.columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.slider(f"{col}", min_val, max_val, mean_val)

    # Prediksi
    input_df = pd.DataFrame([input_data])
    pred_result = model.predict(input_df)[0]
    st.write(f"### Hasil Prediksi Risiko Insomnia: {pred_result:.2f}")

    if pred_result < 1.0:
        st.success("Kategori Risiko: Rendah")
    elif pred_result < 2.0:
        st.warning("Kategori Risiko: Sedang")
    else:
        st.error("Kategori Risiko: Tinggi")
