"""
TCN-based model
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


class TCN(ModelInterfaceDL):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)

        self.parameter_list = {
                              'conv_filter': [16, 32, 64],
                               'conv_kernel': [3, 5, 7],
                               'conv_activation': ['relu', 'tanh'],
                               'dilation_rate': [1, 2, 4, 8],
                                'dropout_rate': [0.0 , 0.05 , 0.1],
                               'dense_dim': [16, 32, 64],
                               'dense_activation': ['relu', 'elu', 'selu', 'tanh'],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [256, 512, 1024],
                               'epochs': [200],
                               'patience': [20],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'momentum': [0.9, 0.99],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }
        """dict: Dictionary of hyperparameters search space"""
        self.p = {
                'conv_filter': 32,
                  'conv_kernel': 3,
                  'conv_activation': 'relu',
                  'dropout_rate': 0.5,
                  'dilation_rate': 1,
                   'dense_dim': 16,
                  'dense_activation': 'relu',
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
        Create an instance of the model. This function contains the definition and the library of$
        :return: None
        """
        input_shape = self.ds.X_train.shape[1:]

        self.temp_model = Sequential([
            tf.keras.layers.Conv1D(filters=self.p['conv_filter'], kernel_size=self.p['conv_kernel'],
                                   padding="causal", dilation_rate = self.p['dilation_rate'],
                                   activation=self.p['conv_activation'],
                                   input_shape=input_shape),
            tf.keras.layers.Conv1D(filters=self.p['conv_filter'], kernel_size=self.p['conv_kernel'],
                                    padding="causal", dilation_rate = self.p['dilation_rate'],
                                   activation=self.p['conv_activation']),
            tf.keras.layers.Conv1D(filters=self.p['conv_filter'], kernel_size=self.p['conv_kernel'],
                                   padding="causal", dilation_rate = self.p['dilation_rate'],
                                   activation=self.p['conv_activation']),
            tf.keras.layers.Conv1D(filters=self.p['conv_filter'], kernel_size=self.p['conv_kernel'],
                                   padding="causal", dilation_rate = self.p['dilation_rate'],
                                   activation=self.p['conv_activation']),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate = self.p['dropout_rate']),
            tf.keras.layers.Dense(self.p['dense_dim'], activation=self.p['dense_activation']),
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

