import streamlit as st
import pandas as pd
import joblib
from preprocess import load_data
from model import train_model
import os

st.set_page_config(page_title="Battery Cell Energy Forecasting", layout="centered")

st.title("🔋 Energy Consumption Forecasting")

if "df" not in st.session_state:
    if os.path.exists("energy_data.csv"):
        st.session_state.df = load_data("energy_data.csv")
    else:
        st.error("No dataset found!")

if st.checkbox("Show raw dataset"):
    st.write(st.session_state.df.head())

if st.button("Train Model"):
    model = train_model(st.session_state.df)
    st.success("✅ Model trained and saved!")

model = joblib.load("energy_model.pkl")

st.markdown("---")
st.subheader("📈 Predict Energy Consumption")

temp = st.slider("Temperature (°C)", 10, 50, 25)
humidity = st.slider("Humidity (%)", 10, 100, 50)
prod_rate = st.slider("Production Rate", 50, 200, 100)
hour = st.slider("Hour of the Day", 0, 23, 12)
day = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

if st.button("Predict"):
    input_data = pd.DataFrame([[temp, humidity, prod_rate, hour, day]],
                              columns=['temperature', 'humidity', 'production_rate', 'hour', 'dayofweek'])
    pred = model.predict(input_data)[0]
    st.success(f"⚡ Predicted Energy Consumption: {pred:.2f} units")
