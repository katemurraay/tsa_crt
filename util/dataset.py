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
        self.name = filename
        "string: name pof the experiment"

        self.X = []
        """list: Full dataset features in windowed format"""
        self.y = []
        """list: Full dataset labels in windowed format"""
        self.X_array = []
        """list: Full dataset features in series format """
        self.y_array = []
        """list: Full dataset labels in series format """
        self.X_train = []
        """list: Training features in windowed format"""
        self.X_test = []
        """list: Test features in windowed format"""
        self.y_train = []
        """list: Training labels in windowed format"""
        self.y_test = []
        """list: Test labels in windowed format"""
        self.X_train_array = []
        """list: Training features in series format"""
        self.y_train_array = []
        """list: Training labels in series format"""
        self.X_test_array = []
        """list: Test features in series format"""
        self.y_test_array = []
        """list: Test labels in series format"""

        self.training_features = training_features
        """list of strings: columns names of the features for the training"""
        self.target_name = target_name
        """list of strings: Columns names of the labels to predict"""
        self.channels = len(self.training_features)
        """int: number of input dimensions"""

        # Input files
        self.data_file = filename
        """string: dataset name"""
        self.data_path = './saved_data/'
        """string: directory path of the dataset"""

        self.train_split_factor = train_split_factor
        """float: training/Test split factor"""

        self.normalization = None
        """list of strings: list of normalization methods to apply to features columns"""
        self.X_scalers = {}
        """dict: dictionary of scaler used for the features"""
        self.y_scalers = {}
        """dict: dictionary of scaler used for the labels"""

        self.input_window = input_window
        """int:  input sequence, number of timestamps of the time series used for training the model"""
        self.stride = 1
        """int: stride for the windowed dataset creation"""
        self.output_window = output_window
        """int: index of the first future timestamp to predict"""
        self.horizon = horizon
        """int: index of the first future timestamp to predict"""

        self.verbose = 1
        """int: level of verbosity of the dataset operations"""

    def data_save(self, name):
        """
        Save the dataset using pickle package
        :param name: string: name of the output file
        :return: None
        """
        with open(self.data_path + name, 'wb') as file:
            pickle.dump(self, file)
            print("File saved in " + self.data_path + name)

    def data_load(self, name):
        """
        Load the dataset using pickle package
        :param name: string: name of the inout file
        :return: None
        """
        with open(self.data_path + name, 'rb') as file:
            return pickle.load(file)

    def data_summary(self):
        """
        Print a summary of the dataset
        :return: None
        """
        print('Training', self.X_train.shape, 'Testing', self.X_test.shape)

    def dataset_creation(self):
        """
        Create all the datasets components with the training and test sets split.
        :return: None
        """
        if self.data_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")

        # read the csv file into a pandas dataframe
        df = pd.read_csv(self.data_path + self.data_file)

        # windowed dataset creation
        columns = df[self.training_features].to_numpy()
        self.X, self.y = self.__windowed_dataset(columns)
        split_value = int(self.X.shape[0] * self.train_split_factor)
        self.y_train = self.y[:split_value]
        self.y_test = self.y[split_value:]
        self.X_train = self.X[:split_value]
        self.X_test = self.X[split_value:]

        # unidimensional dataset creation
        self.X_array = df[self.target_name].to_numpy()
        self.X_array = self.X_array.reshape(-1, 1)
        split_value = int(self.X_array.shape[0] * self.train_split_factor)
        self.y_train_array = self.X_array[self.horizon:self.horizon+split_value-1]
        self.y_test_array = self.X_array[self.horizon+split_value:]
        self.X_train_array = self.X_array[:split_value-1]
        if self.horizon:
            self.X_test_array = self.X_array[split_value: -self.horizon]
        else:
            self.X_test_array = self.X_array[split_value:]

        if self.verbose:
            print("Data size ", self.X.shape)

        if self.verbose:
            print("Training size ", self.X_train.shape, self.X_train_array.shape)
            print("Training labels size", self.y_train.shape, self.y_train_array.shape)
            print("Test size ", self.X_test.shape, self.X_test_array.shape)
            print("Test labels size", self.y_test.shape, self.y_test_array.shape)

    def dataset_normalization(self, methods=["minmax"], scale_range=(0, 1)):
        """
        Normalize the data column according to the specify parameters.
        :param methods: list of strings: normalization methods to apply to each column.
                        Options: ['minmax', 'standard', None], Default = ["minmax"]
        :param scale_range: list of tuples: scale_range for each scaler. Default=(0,1) for each MinMax scaler
        :return: None
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
                    # window dataset
                    self.X_train[:, :, i] = self.X_scalers[i].fit_transform(self.X_train[:, :, i])
                    self.X_test[:, :, i] = self.X_scalers[i].transform(self.X_test[:, :, i])
                    for j, feature in enumerate(self.target_name):
                        if i == self.training_features.index(feature):
                            self.y_train[:, :, j] = self.y_scalers[i].fit_transform(self.y_train[:, :, j])
                            self.y_test[:, :, j] = self.y_scalers[i].transform(self.y_test[:, :, j])
                    # unidimensional dataset
                    self.X_train_array = self.X_scalers[i].fit_transform(self.X_train_array)
                    self.X_test_array = self.X_scalers[i].transform(self.X_test_array)
                    self.y_train_array = self.y_scalers[i].fit_transform(self.y_train_array)
                    self.y_test_array = self.y_scalers[i].transform(self.y_test_array)

    def metadata_creation(self):
        """
        Add metadata to the dataset features. To implement according to the data format.
        :return: None
        """
        pass

    def __windowed_dataset(self, dataset):
        """

        :param dataset: np.array: features of the dataset
        :return: X: np.array: windowed version of the features
                 y: np.array: windowed version of the labels
        """
        dataset = tf.data.Dataset.from_tensor_slices(dataset)
        dataset = dataset.window(self.input_window + self.output_window, stride=self.stride, shift=1,
                                 drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.input_window + self.output_window))
        dataset = dataset.map(lambda window: (window[:-self.output_window], window[-self.output_window:]))
        X, y = [], []
        a = list(dataset.as_numpy_iterator())
        for i, (A, b) in enumerate(a):
            if i == len(a) - self.horizon:
                break
            X.append(A)
            y.append(a[i + self.horizon][1])
        X = np.array(X)
        y = np.array(y)
        indexes = []
        for feature in self.target_name:
            indexes.append(self.training_features.index(feature))
        return X, y[:, :, indexes]
