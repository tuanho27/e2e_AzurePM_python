## Dataset with Feature Engineering
# - What features to create
# - How the features impact model's predictive accuracy
# - Iterating if necessary

from calendar import day_abbr
import glob
from functools import reduce
from ipaddress import ip_address
import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from arize.pandas.logger import Client
from arize.utils.types import Environments, ModelTypes, EmbeddingColumnNames, Schema
from arize.pandas.embeddings.tabular_generators import EmbeddingGeneratorForTabularFeatures
import llama_cpp

seed = 42

class AzurePMDataset(object):
    def __init__(self, datatype='csv', test_ratio=0.2) -> None:
        self.datatype = datatype
        self.num_event = 300000
        self.test_ratio = test_ratio
        self.time_split = False
        self.balancing = True
        self.generator = EmbeddingGeneratorForTabularFeatures(
            model_name="distilbert-base-uncased",
            tokenizer_max_length=16
            )

    def xy_split(self, data):
        data = data.reset_index(drop = True)
        columns_to_drop = ['datetime', 'machineID']
        columns_to_keep = ['volt', 'rotate', 'pressure','vibration']
        # return (data.drop(columns_to_drop, axis=1),
                    # data['failure'])
        fdata = data[columns_to_keep] 
        return fdata, data['failure']

    def create_lstm_feature(self, data, start, end):
        # create features from the selected machine
        pressure = data.loc[start: end, 'pressure']
        timestamp = pd.to_datetime(data.loc[start: end, 'datetime'])
        timestamp_hour = timestamp.map(lambda x: x.hour)
        timestamp_dow = timestamp.map(lambda x: x.dayofweek)

        timestamp_hour_onehot = pd.get_dummies(timestamp_hour).to_numpy()

        # apply min-max scaler to numerical data
        scaler = MinMaxScaler()
        pressure = scaler.fit_transform(np.array(pressure).reshape(-1,1))
        feature = np.concatenate([pressure, timestamp_hour_onehot], axis=1)

        X = feature[:-1]
        y = np.array(feature[5:,0]).reshape(-1,1)

        return X, y, scaler

    def preprocess_for_models(self, filenames, training=False, lstm_data=False, llm_embedding=False):
        # assert len(filenames)!=0, "Cannot find input data!"
        ## Reading input data
        if self.datatype=="csv":
            for i, file_data in enumerate(filenames):
                if "_telemetry" in file_data:
                    telemetry_df = pd.read_csv(file_data)
                    if not training:
                        columns_to_keep = ['volt', 'rotate', 'pressure','vibration']
                        telemetry_df = telemetry_df[columns_to_keep]
                        if not llm_embedding:
                            return telemetry_df
                        else:
                            telemetry_df= telemetry_df[:self.num_event].reset_index(drop=True)
                            embedding_array_train = self.generator.generate_embeddings(
                                telemetry_df,
                                selected_columns  = list(['volt', 'rotate', 'pressure','vibration']),
                            )
                            return np.column_stack((telemetry_df.values, embedding_array_train.tolist()))                        
                if "_errors" in file_data:
                    error_df = pd.read_csv(file_data)
                    error_df['error1']=0
                    error_df['error2']=0
                    error_df['error3']=0
                    error_df['error4']=0
                    error_df['error5']=0
                    for i,eid in enumerate(error_df.errorID):
                        if eid=="error1":
                            error_df.at[i,'error1']=1
                        elif eid=="error2":
                            error_df.at[i,'error2']=1
                        elif eid=="error3":
                            error_df.at[i,'error3']=1
                        elif eid=="error4":
                            error_df.at[i,'error4']=1
                        elif eid=="error5":
                            error_df.at[i,'error5']=1
                    error_df=error_df.drop(columns=['errorID']) 
                if "_failures" in file_data:
                    failure_df = pd.read_csv(file_data)
                    failure_df['failure_comp1']=0
                    failure_df['failure_comp2']=0
                    failure_df['failure_comp3']=0
                    failure_df['failure_comp4']=0
                    for i, fid in enumerate(failure_df.failure):
                        if fid=="comp1":
                            failure_df.at[i,'failure_comp1']=1
                        elif fid=="comp2":
                            failure_df.at[i,'failure_comp2']=1
                        elif fid=="comp3":
                            failure_df.at[i,'failure_comp3']=1
                        elif fid=="comp4":
                            failure_df.at[i,'failure_comp4']=1
                    failure_df=failure_df.drop(columns=['failure']) 
                if "_maint" in file_data:
                    maint_df = pd.read_csv(file_data)
                    maint_df['maint_comp1']=0
                    maint_df['maint_comp2']=0
                    maint_df['maint_comp3']=0
                    maint_df['maint_comp4']=0
                    for i,mid in enumerate(maint_df.comp):
                        if mid=="comp1":
                            maint_df.at[i,'maint_comp1']=1
                        elif mid=="comp2":
                            maint_df.at[i,'maint_comp2']=1
                        elif mid=="comp3":
                            maint_df.at[i,'maint_comp3']=1
                        elif mid=="comp4":
                            maint_df.at[i,'maint_comp4']=1
                    maint_df=maint_df.drop(columns=['comp'])
                if "_machines" in file_data:
                    machine_df = pd.read_csv(file_data)
                    machine_df=machine_df.drop(columns=['model'])                    
            # assert len(filenames)==5, "Not enough input file, please check it!"
            data1 = pd.merge(telemetry_df, error_df, on=['datetime','machineID'], how='left').fillna(0)
            data2 = pd.merge(data1, failure_df, on=['datetime','machineID'], how='left').fillna(0)
            data3 = pd.merge(data2, maint_df, on=['datetime','machineID'], how='left').fillna(0)
            data = pd.merge(data3, machine_df, on=['machineID'], how='left').fillna(0)
            data = data.groupby(['machineID','datetime']).max()
            data=data.reset_index()
            data['failure']=((data['failure_comp1'] + 
                            data['failure_comp2'] + 
                            data['failure_comp3'] + 
                            data['failure_comp2'])/4).astype(dtype=bool)
            data['maint']=((data['maint_comp1'] + 
                            data['maint_comp2'] + 
                            data['maint_comp3'] + 
                            data['maint_comp2'])/4).astype(dtype=bool)
            data['error']=((data['error1'] + 
                            data['error2'] + 
                            data['error3'] + 
                            data['error4'] + 
                            data['error5'])/5).astype(dtype=bool)
            # data['anomaly']=(data['failure'] & data['maint'] & data['error'])
        else:
            for i, file_data in enumerate(filenames):
                if "_data" in file_data:
                    data_df = pd.read_feather(file_data)
                    if not training:
                        columns_to_keep = ['volt', 'rotate', 'pressure','vibration']
                        data_df = data_df[columns_to_keep]
                        if not llm_embedding:
                            return data_df
                        else:
                            data_df= data_df[:self.num_event].reset_index(drop=True)
                            embedding_array_train = self.generator.generate_embeddings(
                                data_df,
                                selected_columns  = list(['volt', 'rotate', 'pressure','vibration']),
                            )
                            return np.column_stack((data_df.values, embedding_array_train.tolist()))       
                if "_labels" in file_data:
                    label_df = pd.read_feather(file_data)
            data_df = data_df.drop(columns=['datetime', 'machineID', 'model','anomaly'])
            data = pd.concat([data_df, label_df], axis=1).fillna(0)
            data = data.groupby(['machineID','datetime']).max()
            data=data.reset_index()

        # data.sample(self.num_event, replace=True)
        # Train/test split strategies (time-dependent, asset ID-based)
        if self.time_split:
            # train, test = train_test_split(data, test_size=self.test_ratio, shuffle=True)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train, test = sss.split(data)
        else:
            # asset ID-based split
            unique_assets = data.reset_index().machineID.unique()
            train_assets, test_assets = train_test_split(
                unique_assets, test_size=self.test_ratio, random_state=seed)
            train = data[data.machineID.isin(train_assets)]
            test = data[data.machineID.isin(test_assets)]

        if lstm_data:
            # Select the date to check from failure records
            st_train = train.loc[train['datetime'] == "2015-11-25"].index.values[0]
            # Filter one month window train data in the history
            start_period = st_train - 30*24
            end_period = st_train + 30*24
            X_train, Y_train, _ = self.create_lstm_feature(train, start_period, end_period)
            
            # Filter one month window test data
            st_test = test.loc[test['datetime'] == "2016-01-01"].index.values[0]
            start_period = st_test - 30*24
            end_period = st_test + 30*24
            X_test, Y_test, _ = self.create_lstm_feature(test, start_period, end_period)
            return X_train, X_test, Y_train, Y_test
        
        X_train, Y_train = self.xy_split(train)
        X_test, Y_test = self.xy_split(test)
        print('Original Train dataset shape %s' % Counter(Y_train))
        print('Original Test dataset shape %s' % Counter(Y_test))

        if self.balancing:
            print("Handle imbalance data by oversampling!")
            all_classes = Counter(Y_train)
            majority_class = all_classes.most_common(1)
            minority_classes = all_classes.most_common()[1:]

            minority_classes_size = sum([c[1] for c in minority_classes])
            desired_minority_classes_size = Y_train.count() * 0.1
            scale = desired_minority_classes_size / minority_classes_size
            ratio = None
            if scale > 1:
                ratio = dict((c[0], int(c[1] * scale)) for c in minority_classes)

            sm = SMOTE(sampling_strategy=ratio, random_state=seed)
            X_train_balance, Y_train_balance = sm.fit_resample(X_train, Y_train)
            print('Resampled train dataset shape %s' % Counter(Y_train_balance))

            X_train = X_train_balance 
            Y_train = Y_train_balance
            
        if llm_embedding:
            print(f"Generate LLM embedding for each row data with {self.num_event} event!")
            X_train= X_train[:self.num_event].reset_index(drop=True)
            embedding_array_train = self.generator.generate_embeddings(
                X_train,
                selected_columns  = list(['volt', 'rotate', 'pressure','vibration']),
                # return_prompt_col = False
            )
            X_test= X_test[:self.num_event].reset_index(drop=True)
            embedding_array_test = self.generator.generate_embeddings(
                X_test,
                selected_columns  = list(['volt', 'rotate', 'pressure','vibration']),
                # return_prompt_col = False
            )
            # Update the embedding features 
            combined_data_train = np.column_stack((X_train.values, embedding_array_train.tolist()))
            combined_data_test = np.column_stack((X_test.values, embedding_array_test.tolist()))
            
            Y_train = Y_train[:self.num_event]
            Y_test = Y_test[:self.num_event]

            return combined_data_train, combined_data_test, Y_train, Y_test

        return X_train, X_test, Y_train, Y_test
