import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras

import pickle
import random
import math


class DatasetInterface:
    def __init__(self, filename="res_task_a.csv", input_window=10, output_window=1, horizon=0, target_names=[],
                 train_split=0.8):
        # Definition of all the instance attributes

        ### COMMON ATTRIBUTES ###
        # Name of the experiment
        self.name = filename

        # Full Dataset
        self.X = []
        self.y = []

        # Training instances
        self.X_train = []
        # Test instances
        self.X_test = []
        # Training labels
        self.y_train = []
        # Test labels
        self.y_test = []

        # Column to predict
        self.target_name = target_names
        self.channels = 1

        # Input files
        self.data_file = filename
        self.data_path = './saved_data/'

        # Train/test split
        self.train_split = train_split

        # Type of  data normalization used
        self.normalization = "minmax" 
        self.scalers = []

        # Input window
        self.window_size = input_window
        self.stride = 1
        self.output_window = output_window
        self.horizon = horizon

        # Configuration
        self.verbose = 1

    def data_save(self, name):
        with open(self.data_path + name, 'wb') as file:
            pickle.dump(self, file)
            print("File saved in " + self.data_path + name)

    def data_load(self, name):
        with open(self.data_path + name, 'rb') as file:
            return pickle.load(file)

    def data_summary(self):
        print('Training', self.X_train.shape, 'Testing', self.X_test.shape)

    def dataset_creation(self):
        if self.data_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")
        df = pd.read_csv(self.data_path + self.data_file)

        mask = df.index < int(df.shape[0] * self.train_split)

        df_train = df[mask]
        df_test = df[~ mask]
        self.train_features = df_train["time"] 
        self.test_features = df_test["time"] 
        self.X = df[self.target_name].to_numpy()
        self.y_train = df_train[self.target_name].to_numpy()
        self.y_test = df_test[self.target_name].to_numpy()
        if self.channels == 1:
            self.X = self.X.reshape(-1, 1)
            self.y_train = self.y_train.reshape(-1, 1)
            self.y_test = self.y_test.reshape(-1, 1)
        self.X_train = df_train[self.target_name].to_numpy()
        self.X_test = df_test[self.target_name].to_numpy()

        split_value = int(self.X.shape[0] * self.train_split)
        self.X, self.y = self.windowed_dataset(self.X)

        self.X_train = self.X[:split_value]
        self.y_train = self.y[:split_value]
        self.X_test = self.X[split_value:]
        self.y_test = self.y[split_value:]

        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)

        if self.verbose:
            print("Data size ", self.X.shape)

        if self.verbose:
            print("Training size ", self.X_train.shape)
            print("Training labels size", self.y_train.shape)

        if self.verbose:
            print("Test size ", self.X_test.shape)
            print("Test labels size", self.y_test.shape)

        # Normalization
        self.scalers = {}
        if self.normalization is not None:
            if self.verbose:
                print("Data normalization")
            for i in range(self.channels):
                if self.normalization == "standard":
                    self.scalers[i] = StandardScaler()
                elif self.normalization == "minmax":
                    self.scalers[i] = MinMaxScaler((-1, 1)) 
                self.X_train[:, :, i] = self.scalers[i].fit_transform(self.X_train[:, :, i])
                self.X_test[:, :, i] = self.scalers[i].transform(self.X_test[:, :, i])
                self.y_train = self.scalers[i].fit_transform(self.y_train)

                self.y_test = self.scalers[i].transform(self.y_test)

        if self.meta is not None:
            if self.verbose:
                print("Metadata")
            if self.meta == "categorical":
                tmp1 = np.zeros(
                    (self.X_train.shape[0] - self.window_size, self.X_train.shape[1] + 2, self.X_train.shape[2]))
                tmp1[:, :-2, :] = self.X_train[self.window_size:]
                tmp1[:, -2, :] = np.reshape(df_train['day_of_week'].iloc[self.window_size:].values,
                                            (self.X_train.shape[0] - self.window_size, 1))
                tmp1[:, -1, :] = np.reshape(df_train['hour_of_day'].iloc[self.window_size:].values,
                                            (self.X_train.shape[0] - self.window_size, 1))
                self.X_train = tmp1

                tmp2 = np.zeros((self.X_test.shape[0], self.X_test.shape[1] + 2, self.X_test.shape[2]))
                tmp2[:, :-2, :] = self.X_test
                tmp2[:, -2, :] = np.reshape(df_test['day_of_week'].iloc[self.window_size:].values,
                                            (self.X_test.shape[0], 1))
                tmp2[:, -1, :] = np.reshape(df_test['hour_of_day'].iloc[self.window_size:].values,
                                            (self.X_test.shape[0], 1))
                self.X_test = tmp2

    def windowed_dataset(self, series):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(self.window_size + 1, stride=self.stride, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size + 1))
        dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

        inputs, targets = [], []
        a = list(dataset.as_numpy_iterator())
        for i, (X, y) in enumerate(a):
            if i == len(a) - self.horizon:
                break
            inputs.append(X)
            targets.append(a[i + self.horizon][1])
        inputs = np.array(inputs)
        targets = np.vstack(targets)
        return inputs, targets
