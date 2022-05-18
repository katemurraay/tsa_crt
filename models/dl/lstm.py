import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.dl.model_interface_dl import ModelInterfaceDL
from datetime import datetime
import pickle


class LSTMPredictor(ModelInterfaceDL):
    def __init__(self):
        ModelInterfaceDL.__init__(self, "LSTMPredictor")
        self.parameter_list = {'first_conv_dim': [32, 64, 128],
                               'first_conv_kernel': [3, 5, 7, 11],
                               'first_conv_activation': ['relu', 'tanh'],
                               'second_lstm_dim': [16, 32, 64],
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
        self.p = {'first_conv_dim': 32,
                  'first_conv_kernel': 5,
                  'first_conv_activation': 'relu',
                  'second_lstm_dim': 16,
                  'first_dense_dim': 16,
                  'first_dense_activation': 'elu',
                  'dense_kernel_init': 'he_normal',
                  'batch_size': 256,
                  'epochs': 50,
                  'patience': [50],
                  'optimizer': 'adam',
                  'lr': 1E-4,
                  'momentum': 0.9,
                  'decay': 1E-4,
                  }

    def create_model(self):
        self.temp_model = Sequential([
            tf.keras.layers.Conv1D(filters=self.p['first_conv_dim'], kernel_size=self.p['first_conv_kernel'],
                                   strides=1, padding="causal",
                                   activation=self.p['first_conv_activation'],
                                   input_shape=self.input_shape),
            tf.keras.layers.LSTM(self.p['second_lstm_dim']),
            tf.keras.layers.Dense(self.p['first_dense_dim'], activation=self.p['first_dense_activation']),
            tf.keras.layers.Dense(1),
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
        self.save_check = custom_keras.CustomSaveCheckpoint(self)
        self.es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=self.p['patience'])

    def fit(self):
        fit_start = datetime.now()
        history = self.temp_model.fit(self.ds.X_train, self.ds.y_train, epochs=self.p['epochs'],
                                      batch_size=self.p['batch_size'],
                                      validation_split=0.2, verbose=2, callbacks=[self.es, self.save_check])

        self.model = self.save_check.dnn.model

        fit_time = datetime.now() - fit_start
        return fit_time

    def tuning(self, X, y):
        tuning_start = datetime.now()
        self.temp_model.fit(X, y, epochs=self.p['epochs'], batch_size=self.p['batch_size'],
                            validation_split=0.2, verbose=2, callbacks=[self.es, self.save_check])

        tuning_time = datetime.now() - tuning_start
        self.model = self.save_check.dnn.model

        return self.model, tuning_time

    def predict(self, X):
        predict_start = datetime.now()
        prediction = self.model.predict(X)
        prediction = prediction[:, -1]
        prediction = prediction[~np.isnan(prediction)]
        predict_time = datetime.now() - predict_start

        return prediction, predict_time
