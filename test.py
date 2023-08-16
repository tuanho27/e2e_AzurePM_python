import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dataset import AzurePMDataset
import glob
import joblib


# Load XGBoost model
xgb_model = XGBClassifier()
xgb_model.load_model("./weights/model_x.json")
print("Load XGBoost model!")

# Load RandomForestClassifier model
boost_llm_model = joblib.load("./weights/model_lm.pkl")
print("Load RandomForestClassifier model!")

# Reading data for training
data_dir = './data'
filenames = glob.glob(data_dir + '/origin/*.csv')

dataset = AzurePMDataset(datatype="csv")

sensor_data_description = dataset.preprocess_for_models(filenames, training=False)
sensor_data_llm_description = dataset.preprocess_for_models(filenames, training=False, llm_embedding=True)

# Call prediction functions
xgb_result = xgb_model.predict(sensor_data_description)
print(xgb_result)
boost_llm_result = boost_llm_model.predict(sensor_data_llm_description)
print(boost_llm_result)
