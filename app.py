import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Page config
st.set_page_config(page_title="🌦️ Rainfall Prediction", layout="wide")
st.markdown("<h1 style='text-align: center; color: #6a89cc;'>Rainfall Prediction App</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("rain_model.pkl")  # Make sure model.pkl is present

model = load_model()

# Sidebar inputs
with st.sidebar:
    st.header("🧾 Manual Weather Input")

    pressure = st.number_input("Pressure", min_value=900, max_value=1100, value=1010)
    dewpoint = st.number_input("Dew Point", min_value=0, max_value=40, value=10)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    cloud = st.slider("Cloud (%)", 0, 100, 40)
    sunshine = st.slider("Sunshine (hrs)", 0.0, 12.0, 6.0)
    wind_dir = st.slider("Wind Direction (°)", 0, 360, 180)
    wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 10)

    if st.button("🌧️ Predict Manually"):
        input_data = pd.DataFrame([[
            pressure, dewpoint, humidity, cloud,
            sunshine, wind_dir, wind_speed
        ]], columns=[
            "pressure", "dewpoint", "humidity ",
            "cloud ", "sunshine", "         winddirection", "windspeed"
        ])

        pred = model.predict(input_data)[0]
        label = "🌧️ Rain Expected" if pred == 1 else "☀️ No Rain"
        st.success(f"Prediction: {label}")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV with 7 features", type=["csv"])

# Main layout
left, right = st.columns([1, 1.2])

with left:
    st.subheader("📋 Uploaded Data Preview")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        expected_cols = [
            "pressure", "dewpoint", "humidity ",
            "cloud ", "sunshine", "         winddirection", "windspeed"
        ]

        if all(col in df.columns for col in expected_cols):
            prediction = model.predict(df[expected_cols])
            df["Prediction"] = np.where(prediction == 1, "Rain 🌧️", "No Rain ☀️")
            st.dataframe(df)
        else:
            st.error("CSV file does not contain all required columns.")

with right:
    st.subheader("📊 Rainfall Prediction Charts")

    if uploaded_file is not None and "Prediction" in df.columns:
        counts = df["Prediction"].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sns.set_palette(["#a2d5f2", "#ffc3a0"])  # Pastel tones

        # Bar chart
        counts.plot(kind="bar", ax=ax1)
        ax1.set_title("Rain vs No Rain (Bar)", fontsize=14)
        ax1.set_ylabel("Count")

        # Pie chart
        counts.plot(kind="pie", ax=ax2, autopct='%1.1f%%', startangle=90)
        ax2.set_ylabel("")
        ax2.set_title("Rain vs No Rain (Pie)", fontsize=14)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Upload a valid CSV file to see visualizations.")
