"""
LSTM-based model
Inherits from ModelInterfaceDL class
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.dl.model_interface_dl import ModelInterfaceDL
from datetime import datetime
import pickle


class LSTM(ModelInterfaceDL):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)

        self.parameter_list = {'first_conv_dim': [32, 64, 128],
                               'first_conv_kernel': [3, 5, 7, 11],
                               'first_conv_activation': ['relu', 'tanh'],
                               'first_lstm_dim': [16, 32, 64],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': ['relu', 'elu', 'selu', 'tanh'],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [50],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'momentum': [0.9, 0.99],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }
        """dict: Dictionary of hyperparameters search space"""

        self.p = {'first_conv_dim': 32,
                  'first_conv_kernel': 5,
                  'first_conv_activation': 'relu',
                  'first_lstm_dim': 16,
                  'first_dense_dim': 16,
                  'first_dense_activation': 'relu',
                  'dense_kernel_init': 'he_normal',
                  'batch_size': 256,
                  'epochs': 1000,
                  'patience': 50,
                  'optimizer': 'adam',
                  'lr': 1E-4,
                  'momentum': 0.9,
                  'decay': 1E-4,
                  }
        """dict: Dictionary of hyperparameter configuration of the model"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        input_shape = self.ds.X_train.shape[1:]

        self.temp_model = Sequential([
            tf.keras.layers.Conv1D(filters=self.p['first_conv_dim'], kernel_size=self.p['first_conv_kernel'],
                                   strides=1, padding="causal",
                                   activation=self.p['first_conv_activation'],
                                   input_shape=input_shape),
            tf.keras.layers.LSTM(self.p['first_lstm_dim']),
            tf.keras.layers.Dense(self.p['first_dense_dim'], activation=self.p['first_dense_activation']),
            tf.keras.layers.Dense(self.ds.y_train.shape[2]),
        ])

        if self.p['optimizer'] == 'adam':
            opt = Adam(learning_rate=self.p['lr'], decay=self.p['decay'])
        elif self.p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=self.p['lr'], momentum=self.p['momentum'])
        self.temp_model.compile(loss='mean_squared_error',
                                optimizer=opt,
                                metrics=["mse", "mae"])
