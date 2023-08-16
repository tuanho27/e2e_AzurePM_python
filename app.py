import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model.baseline import LSTMModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from flask import jsonify
import requests
from dataset import AzurePMDataset

def main():
    dataset = AzurePMDataset(datatype="csv")
    # Load XGBoost model
    xgb_model = XGBClassifier()
    xgb_model.load_model("./weights/model_x.json")
    print("Load XGBoost model!")

    # Load RandomForestClassifier model
    boost_llm_model = joblib.load("./weights/model_lm.pkl")
    print("Load RandomForestClassifier model!")

    st.title("Equipment Failure Prediction")
    # Get user input
    st.title('Sensor Data Processing')
    uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

    if uploaded_files is not None:
        sensor_data_description = dataset.preprocess_for_models(uploaded_files, training=False)
        sensor_data_llm_description = dataset.preprocess_for_models(uploaded_files, training=False, llm_embedding=True)

    if len(uploaded_files) == 0:
        return jsonify({"message": "No sensor data available for testing"})

    if st.button("Predict"):
        if sensor_data_description:
            # Call prediction functions
            xgb_result = xgb_model.predict(sensor_data_description)
            boost_llm_result = boost_llm_model.predict(sensor_data_llm_description)
            # response = requests.post("http://localhost:8501", json={"description": sensor_data_description})
            # prediction = response.json()["prediction"]
            st.write("XGBoost Prediction:", xgb_result)
            st.write("Boosting with LLM Prediction:", boost_llm_result)

if __name__ == "__main__":
    main()

