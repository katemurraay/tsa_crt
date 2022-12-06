"""
LSTM & GRU hybrid model based on work of Patel et al. (2020)
Inherits from ModelInterfaceDL class
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.dl.model_interface_dl import ModelInterfaceDL
from datetime import datetime
import pickle



class LSTM_GRU(ModelInterfaceDL):

    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.parameter_list = {
                                'first_lstm_dim': [30, 50, 75],
                                'lstm_activation': ['relu', 'tanh'],
                                'first_dropout_rate': [0.0, 0.05, 0.1],
                                'second_lstm_dim': [30, 50, 75],
                                'first_dense_dim': [16, 32, 64],
                               'dense_activation': ['relu', 'elu', 'selu', 'tanh'],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'gru_dim':[30,50,75],
                                'gru_activation': ['relu', 'tanh'],
                               'second_dropout_rate': [0.0, 0.05, 0.1],
                                'second_dense_dim': [16, 32, 64],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [50],
                               'optimizer': ['adam', 'nadam', 'rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'momentum': [0.9, 0.99],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }
        """dict: Dictionary of hyperparameters search space"""

        self.p = {
                'first_lstm_dim': 50,
                'lstm_activation': 'relu',
                'first_dropout_rate': 0.05, 
                'second_lstm_dim':  50,
                'first_dense_dim':  32,
                'dense_activation': 'relu',
                'dense_kernel_init': 'he_normal',
                'gru_dim':50,
                'gru_activation':'relu',
                'second_dropout_rate':  0.05,
                'second_dense_dim':  32,
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
        

        input = tf.keras.Input(shape=(None, self.ds.X_train.shape[2]))


        #LSTM 
        x = tf.keras.layers.LSTM(self.p['first_lstm_dim'], activation = self.p['lstm_activation'], return_sequences = True)(input)
        x = tf.keras.layers.Dropout(rate= self.p['first_dropout_rate'])(x)
        x = tf.keras.layers.LSTM(self.p['second_lstm_dim'], activation = self.p['lstm_activation'])(x)
        lstm_model = tf.keras.layers.Dense(self.p['first_dense_dim'], activation = self.p['dense_activation'])(x)


        #GRU
        y = tf.keras.layers.GRU(self.p['gru_dim'], activation = self.p['gru_activation'])(input)
        y = tf.keras.layers.Dropout(rate = self.p['second_dropout_rate'])(y)
        gru_model = tf.keras.layers.Dense(self.p['second_dense_dim'], activation = self.p['dense_activation'])(y)


        concatenated = tf.keras.layers.concatenate([lstm_model, gru_model])
        output = tf.keras.layers.Dense(self.ds.y_train.shape[2])(concatenated)
        self.temp_model = tf.keras.Model(inputs = input, outputs = output)


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

