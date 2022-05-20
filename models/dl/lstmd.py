"""
LSTMD(istribution) model
Inherits from ModelProbabilisticDL class
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import talos
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Activation, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.dl.model_probabilistic_dl import ModelProbabilisticDL
from sklearn.metrics import mean_squared_error
from util import plot_training
import tensorflow_probability as tfp
import os
import pandas as pd
from datetime import datetime

import pickle
from models.model_probabilistic import ModelProbabilistic


class LSTMD(ModelProbabilisticDL):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)

        self.temp_model = None
        self.parameter_list = {'first_conv_dim': [32, 64, 128],
                               'first_conv_kernel': [3, 5, 7, 11],
                               'first_conv_activation': ['relu'],
                               'second_lstm_dim': [16, 32, 64],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': [keras.activations.relu],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [50],
                               'optimizer': ['adam', 'rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'momentum': [0.9],
                               'decay': [1E-3, 1E-4]
                               }
        """dict: Dictionary of hyperparameters search space"""
        self.p = {'first_conv_dim': 32,
                  'first_conv_activation': 'relu',
                  'first_conv_kernel': 7,
                  'first_lstm_dim': 16,
                  'first_dense_dim': 16,
                  'first_dense_activation': keras.activations.relu,
                  'batch_size': 256,
                  'epochs': 5,
                  'patience': 50,
                  'optimizer': 'adam',
                  'batch_normalization': True,
                  'lr': 1E-4,
                  'momentum': 0.99,
                  'decay': 1E-4,
                  }
        """dict: Dictionary of hyperparameter configuration of the model"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        tf.keras.backend.clear_session()
        input_shape = self.ds.X_train.shape[1:]

        input_tensor = Input(shape=input_shape)

        x = Conv1D(filters=self.p['first_conv_dim'], kernel_size=self.p['first_conv_kernel'],
                   strides=1, padding="causal",
                   activation=self.p['first_conv_activation'],
                   input_shape=input_shape)(input_tensor)

        x = LSTM(self.p['first_lstm_dim'])(x)

        x = layers.Dense(self.p['first_dense_dim'], activation=self.p['first_dense_activation'])(x)

        distribution_params = layers.Dense(units=2 * self.ds.y_train.shape[2])(x)
        outputs = tfp.layers.IndependentNormal(self.ds.y_train.shape[2])(distribution_params)

        self.temp_model = Model(inputs=input_tensor, outputs=outputs)

        if self.p['optimizer'] == 'adam':
            opt = Adam(learning_rate=self.p['lr'], decay=self.p['decay'])
        elif self.p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=self.p['lr'])
        elif self.p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=self.p['lr'], momentum=self.p['momentum'])
        self.temp_model.compile(loss=self.__negative_loglikelihood,
                                optimizer=opt,
                                metrics=["mse", "mae"])

    def __negative_loglikelihood(self, targets, estimated_distribution):
        """
        Negative log-likelihood custom function
        :param targets: np.array: labels
        :param estimated_distribution: np.array: prediction
        :return:
        """
        return -estimated_distribution.log_prob(targets)
