# # Model Training
# Formulate a multi-class classification problem as follows:
import argparse
from ast import Raise
import glob
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from model.baseline import LSTMModel
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
from dataset import AzurePMDataset
from utils import plot_confusion_matrix

seed = 42

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    parser.add_argument('--datatype',
                        help='input file type',
                        default="feather",
                        type=str)
    parser.add_argument('--model',
                        help="r: randomforest, x: XGboost, l:LSTM, lm: LLM",
                        default="lm",
                        type=str)
    parser.add_argument('--resume',
                        help="weight load",
                        default=None,
                        type=str)
    parser.add_argument('--log',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Reading data for training
    data_dir = './data'
    if args.datatype=="csv":
        filenames = glob.glob(data_dir + '/origin/*.csv')
    else:
        filenames = glob.glob(data_dir + '/*.feather')

    dataset = AzurePMDataset(datatype=args.datatype)

    ## Model training
    if args.model == "x":
        print("Train model with XGBClassifier")
        X_train, X_test, Y_train, Y_test = dataset.preprocess_for_models(filenames, training=True)
        eval_set = [(X_test, Y_test)]

        model = XGBClassifier(random_state=seed)
        model.fit(X_train, Y_train, eval_set=eval_set, verbose=True)
        Y_predictions = model.predict(X_test)
        ## Save the model and test sample
        model.save_model(f'weights/model_{args.model}.json') 

    elif args.model == 'r':
        print("Train model with RandomForestClassifier")
        X_train, X_test, Y_train, Y_test = dataset.preprocess_for_models(filenames, training=True)
        model = RandomForestClassifier(random_state=seed, verbose=1)
        model.fit(X_train, Y_train)
        Y_predictions = model.predict(X_test)
        joblib.dump(model, f'weights/model_{args.model}.pkl') 

    elif args.model=="l":
        print("Train model with LSTM with previous 1 month data")
        X_train, X_test, Y_train, Y_test = dataset.preprocess_for_models(filenames, training=True, lstm_data=True)
        input_size = X_train.shape[1]
        hidden_size = 32
        output_size = 1
        num_layers = 2

        lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)

        for epoch in range(100):
            optimizer.zero_grad()
            outputs = lstm_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            print(f"[Epoch]: {epoch},  Loss: {loss}")
            loss.backward()
            optimizer.step()

        lstm_pred_tensor = lstm_model(torch.tensor(X_test, dtype=torch.float32))
        Y_predictions = lstm_pred_tensor.detach().numpy()
        torch.save(lstm_model.state_dict(),f"weights/model_{args.model}.pth")

    elif args.model == "lm":
        print("Train model Leveraging LLMs for maintenance prediction")
        X_train, X_test, Y_train, Y_test = dataset.preprocess_for_models(filenames, training=True, llm_embedding=True)
        # clf.fit(X_train, Y_train)
        # Initialize LLM for maintenance insights (similar to previous examples)
        eval_set = [(X_test, Y_test)]
        model = RandomForestClassifier(random_state=seed, verbose=1)
        model.fit(X_train, Y_train)
        Y_predictions = model.predict(X_test)
        joblib.dump(model, f'weights/model_{args.model}.pkl') 

    else:
        raise ValueError("Model invalid!")

    # # ## Model evaluation
    print(classification_report(Y_test, Y_predictions, digits=4))
    print('Accuracy: ', accuracy_score(Y_test, Y_predictions))

    if args.log:        
        binarizer = LabelBinarizer()
        binarizer.fit(Y_train)
        def auc_score(y_true, y_pred):
            return roc_auc_score(binarizer.transform(y_true), binarizer.transform(y_pred), average='macro')
        print('ROC AUC scores: ',auc_score(Y_test, Y_predictions))
        
        cm = confusion_matrix(Y_test, binarizer.inverse_transform(Y_predictions))
        cm = confusion_matrix(Y_test, Y_predictions)
        plot_confusion_matrix(cm, ['None'])

    print('Done Training!')

if __name__ == '__main__':
    main()