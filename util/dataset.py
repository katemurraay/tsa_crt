import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
import pickle


class DatasetInterface:
    def __init__(self, filename="", input_window=10, output_window=1, horizon=0, target_name=[],
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

        # Timeseries version for stat techniques
        self.X_train_series = []
        self.y_train_series = []
        self.X_test_series = []
        self.y_test_series = []

        # Column to predict
        self.target_name = target_name
        self.channels = len(self.target_name)

        # Input files
        self.data_file = filename
        self.data_path = './saved_data/'

        # Train/test split
        self.train_split = train_split

        # Type of  data normalization used
        self.normalization = None
        self.X_scalers = {}
        self.y_scalers = {}

        # Input window
        self.window_size = input_window
        self.stride = 1
        self.output_window = output_window
        self.horizon = horizon

        # Configuration
        self.verbose = 1

        ### SPECIFIC CONFIGURATION ##

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
        # To be implemented by the specific class
        if self.data_file is None:
            print("ERROR: Source files not specified")
            return
        if self.verbose:
            print("Data load")
        df = pd.read_csv(self.data_path + self.data_file)

        self.X = df[self.target_name].to_numpy()
        if self.channels == 1:
            self.X = self.X.reshape(-1, 1)

        split_value = int(self.X.shape[0] * self.train_split)
        self.X_train_series = self.X[:split_value]
        self.y_train_series = self.y[:split_value]
        self.X_test_series = self.X[split_value:]
        self.y_test_series = self.y[split_value:]

        self.X, self.y = self.__windowed_dataset(self.X)
        print(self.X.shape, self.y.shape)
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

    def dataset_normalization(self, method="minmax", scale_range=(0, 1)):
        # Normalization
        self.X_scalers = {}
        self.y_scalers = {}
        if method is not None:
            self.normalization = method
        if self.verbose:
            print("Data normalization")

        for i in range(self.channels):
            if self.normalization == "standard":
                self.X_scalers[i] = StandardScaler()
                self.y_scalers[i] = StandardScaler()
            elif self.normalization == "minmax":
                self.X_scalers[i] = MinMaxScaler(scale_range)
                self.y_scalers[i] = MinMaxScaler(scale_range)
            self.X_train[:, :, i] = self.X_scalers[i].fit_transform(self.X_train[:, :, i])
            self.X_test[:, :, i] = self.X_scalers[i].transform(self.X_test[:, :, i])
            self.y_train[:, :, i] = self.y_scalers[i].fit_transform(self.y_train[:, :, i])
            self.y_test[:, :, i] = self.y_scalers[i].transform(self.y_test[:, :, i])

    def metadata_creation(self):
        pass

    def __windowed_dataset(self, series):
        dataset = tf.data.Dataset.from_tensor_slices(series)
        dataset = dataset.window(self.window_size + self.output_window, stride=self.stride, shift=1,
                                 drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.window_size + self.output_window))
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
        return inputs, targets
