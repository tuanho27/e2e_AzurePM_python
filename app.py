import streamlit as st
import pandas as pd
import numpy as np
import joblib
from model.baseline import LSTMModel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from flask import jsonify
import requests
from arize.pandas.embeddings.tabular_generators import EmbeddingGeneratorForTabularFeatures

def preprocess_for_models(df, llm_embedding=False):
    num_event = 100000
    generator = EmbeddingGeneratorForTabularFeatures(
                model_name="distilbert-base-uncased",
                tokenizer_max_length=16
                )
    columns_to_keep = ['volt', 'rotate', 'pressure','vibration']
    df = df[columns_to_keep]
    df= df[:num_event].reset_index(drop=True)
    embedding_array_train = generator.generate_embeddings(
        df,
        selected_columns  = list(['volt', 'rotate', 'pressure','vibration']),
    )
    return df, np.column_stack((df.values, embedding_array_train.tolist()))

def main():
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

    if uploaded_files:
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files, start=1):
            st.write("Uploaded File:", file.name)
            df = pd.read_csv(file)
            st.write("Preview:")
            sensor_data, sensor_data_llm = preprocess_for_models(df, llm_embedding=True)
            st.write(sensor_data.head())
            progress_bar.progress(int((idx / len(uploaded_files)) * 100))

    if len(uploaded_files) == 0:
        return st.write("No sensor data available for testing")

    if st.button("Predict"):
        # Call prediction functions
        xgb_result = xgb_model.predict(sensor_data)
        boost_llm_result = boost_llm_model.predict(sensor_data_llm)
        # # response = requests.post("http://localhost:8501", json={"description": sensor_data_description})

        st.write("Model Inference Result")
        col1, col2 = st.columns(2)
        with col1:
            st.write("XGBoost Potential Failure Prediction:  ", round(np.mean(xgb_result), 6))
     
        with col2:
            st.write("Boosting & LLM Potential Failure Prediction:",  round(np.mean(boost_llm_result.astype(int)),6))


if __name__ == "__main__":
    main()

