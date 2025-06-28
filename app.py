# importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set Streamlit page configuration
st.set_page_config(page_title="Daily Temperature Forecast App 🌡️", layout="centered")

# Title and description
st.title("Daily Temperature Forecast App 🌤️")
st.markdown("This app predicts the **next day's minimum temperature** based on the past 30 days of data.")

# Load model and scaler
try:
    model = tf.keras.models.load_model("Temp_Predictor_Model.h5", compile=False)
    scaler = joblib.load("Scaler.pkl")
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Upload the CSV file
uploaded_file = st.file_uploader("📁 Upload recent temperature data (CSV with 'Date' and 'Temp')", type=["csv"])

if uploaded_file:
    try:
        # Load and clean the data
        df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
        df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
        df = df.dropna()

        # Normalize data
        data_scaled = scaler.transform(df["Temp"].values.reshape(-1, 1))

        # Check if we have enough data
        if len(data_scaled) < 30:
            st.warning("Please upload at least 30 days of temperature data.")
        else:
            # Use last 30 values for prediction
            last_sequence = data_scaled[-30:].reshape(1, 30, 1)
            prediction_scaled = model.predict(last_sequence)
            prediction_scaled = np.clip(prediction_scaled, 0, 1)
            predicted_temp = scaler.inverse_transform(prediction_scaled)

            st.subheader("🌡️ Predicted Minimum Temperature for Tomorrow:")
            st.success(f"{predicted_temp[0][0]:.2f} °C")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload the temperature CSV file to begin.")
