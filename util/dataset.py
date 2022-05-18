"""
Interface of a Dataset class with shared functionalities
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pickle


class DatasetInterface:
    def __init__(self, filename="", input_window=10, output_window=1, horizon=0, training_features=[], target_name=[],
                 train_split_factor=0.8):
        """
        Constructor of the DatasetInterface class
        :param filename: string: path of the dataset in .csv format. Default = ""
        :param input_window: int: input sequence, number of timestamps of the time series used for training the model
        :param output_window: int: output sequence, length of the prediction. Default = 1 (one-step-ahead prediction)
        :param horizon: int: index of the first future timestamp to predict. Default = 0
        :param training_features: array of string: names of the features used for the training. Default = []
        :param target_name: array of strings: names of the column to predict. Default = []
        :param train_split_factor: float: Training/Test split factor Default = 0.8
        """
        # Common attributes
        self.name = filename
        "string: name pof the experiment"

        self.X = []
        """list: Full dataset features"""
        self.y = []
        """list: Full dataset labels"""
        self.X_train = []
        """list: Training features"""
        self.X_test = []
        """list: Test features"""
        self.y_train = []
        """list: Training labels"""
        self.y_test = []
        """list: Test labels"""

        self.training_features = training_features
        # Column to predict
        self.target_name = target_name
        self.channels = len(self.training_features)

        # Input files
        self.data_file = filename
        self.data_path = './saved_data/'

        # Train/test split
        self.train_split_factor = train_split_factor

        # Type of  data normalization used
        self.normalization = None
        self.X_scalers = {}
        self.y_scalers = {}

        # Input window
        self.input_window = input_window
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
        """
        Create all the datasets components with the training and test sets split.
        :return: None
        """
        # To be implemented by the specific class
        if self.data_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")
        df = pd.read_csv(self.data_path + self.data_file)

        # windowed dataset creation
        if self.input_window > 1:
            columns = df[self.training_features].to_numpy()
            self.X, self.y = self.__windowed_dataset(columns)
            split_value = int(self.X.shape[0] * self.train_split_factor)
            self.y_train = self.y[:split_value]
            self.y_test = self.y[split_value:]
            self.X_train = self.X[:split_value]
            self.X_test = self.X[split_value:]
        else:
            self.X = df[self.target_name].to_numpy()
            self.X = self.X.reshape(-1, 1)
            split_value = int(self.X.shape[0] * self.train_split_factor)
            self.y_train = self.X[:split_value]
            self.y_test = self.X[split_value:]
            self.X_train = self.X[:split_value]
            self.X_test = self.X[split_value:]

        if self.verbose:
            print("Data size ", self.X.shape)

        if self.verbose:
            print("Training size ", self.X_train.shape)
            print("Training labels size", self.y_train.shape)
            print("Test size ", self.X_test.shape)
            print("Test labels size", self.y_test.shape)

    def dataset_normalization(self, methods=["minmax"], scale_range=(0, 1)):
        """

        :param methods:
        :param scale_range:
        :return:
        """
        if self.verbose:
            print("Data normalization")
        if methods is not None and self.channels != len(methods):
            print("ERROR: You have to specify a scaling method for each feature")
            exit(1)
        self.X_scalers = {}
        self.y_scalers = {}
        if methods is not None:
            self.normalization = methods
            for i in range(self.channels):
                if self.normalization[i] is not None:
                    if self.normalization[i] == "standard":
                        self.X_scalers[i] = StandardScaler()
                        self.y_scalers[i] = StandardScaler()
                    elif self.normalization[i] == "minmax":
                        self.X_scalers[i] = MinMaxScaler(scale_range)
                        self.y_scalers[i] = MinMaxScaler(scale_range)
                    if self.input_window > 1:
                        self.X_train[:, :, i] = self.X_scalers[i].fit_transform(self.X_train[:, :, i])
                        self.X_test[:, :, i] = self.X_scalers[i].transform(self.X_test[:, :, i])
                        for j, feature in enumerate(self.target_name):
                            if i == self.training_features.index(feature):
                                self.y_train[:, :, j] = self.y_scalers[i].fit_transform(self.y_train[:, :, j])
                                self.y_test[:, :, j] = self.y_scalers[i].transform(self.y_test[:, :, j])
                    else:
                        self.X_train = self.X_scalers[i].fit_transform(self.X_train)
                        self.X_test = self.X_scalers[i].transform(self.X_test)
                        self.y_train = self.y_scalers[i].fit_transform(self.y_train)
                        self.y_test = self.y_scalers[i].transform(self.y_test)
            # if not self.windowed_creation:
            #     self.X_train = self.X_train.reshape(-1, 1)
            #     self.X_test = self.X_test.reshape(-1, 1)

    def metadata_creation(self):
        """

        :return:
        """
        pass

    def __windowed_dataset(self, dataset):
        """

        :param dataset:
        :return:
        """
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.window(self.input_window + self.output_window, stride=self.stride, shift=1,
                                 drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.input_window + self.output_window))
        dataset = dataset.map(lambda window: (window[:-self.output_window], window[-self.output_window:]))
        inputs, targets = [], []
        a = list(dataset.as_numpy_iterator())
        for i, (X, y) in enumerate(a):
            if i == len(a) - self.horizon:
                break
            inputs.append(X)
            targets.append(a[i + self.horizon][1])
        inputs = np.array(inputs)
        targets = np.array(targets)
        indexes = []
        for feature in self.target_name:
            indexes.append(self.training_features.index(feature))
        return inputs, targets[:, :, indexes]
