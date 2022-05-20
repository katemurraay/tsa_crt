import numpy as np
import tensorflow as tf
import talos
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, Flatten, Activation, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
from models.model_interface import ModelInterface
from sklearn.metrics import mean_squared_error
from datetime import datetime
import pickle


class LSTMPredictor(ModelInterface):
    def __init__(self):
        ModelInterface.__init__(self, "LSTMPredictor")
        self.train_model = None
        self.input_shape = None
        self.model = None
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

    def training(self, X_train, y_train, X_test, y_test, p):
        training_start = datetime.now()
        history, self.model = self.talos_model(X_train, y_train, X_test, y_test, p)
        training_time = datetime.now() - training_start

        self.train_model.summary()
        print(history)

        inference_start = datetime.now()
        forecast = self.train_model.predict(X_test)
        forecast = forecast[:, -1]
        forecast = forecast[~np.isnan(forecast)]
        inference_time = (datetime.now() - inference_start) / y_test.shape[0]

        return self.model, history, forecast, training_time, inference_time

    def tuning(self, X_tuning, y_tuning, p):
        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        tuning_start = datetime.now()
        history = self.train_model.fit(X_tuning, y_tuning, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        tuning_time = datetime.now() - tuning_start
        self.model = save_check.dnn.model

        return self.model, history, tuning_time

    def training_talos(self, X_train, y_train, X_test, y_test, p):
        p = self.parameter_list
        tf.keras.backend.clear_session()
        self.input_shape = X_train.shape[1:]

        t = talos.Scan(x=X_train,
                       y=y_train,
                       model=self.talos_model,
                       experiment_name=self.name,
                       params=p,
                       clear_session=True,
                       print_params=True,
                       round_limit=500)

        return t, None, None

    def load_and_tune(self, X_train, y_train, X_test, y_test, p):
        global opt
        self.train_model = self.load_model(X_train, y_train, X_test, y_test, p)
        self.model = self.train_model

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss='mean_squared_error',
                                 optimizer=opt,
                                 metrics=["mse", "mae"])

        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        forecast = self.train_model.predict(X_test)

        forecast = forecast[:, -1]
        forecast = forecast[~np.isnan(forecast)]

        return self.model, history, forecast

    def load_model(self, X_train, y_train, x_val, y_val, p):
        tf.keras.backend.clear_session()
        input_shape = X_train.shape[1:]
        self.train_model = Sequential([
            tf.keras.layers.Conv1D(filters=p['first_conv_dim'], kernel_size=p['first_conv_kernel'],
                                   strides=1, padding="causal",
                                   activation=p['first_conv_activation'],
                                   input_shape=input_shape),
            tf.keras.layers.LSTM(p['second_lstm_dim']), 
            tf.keras.layers.Dense(p['first_dense_dim'], activation=p['first_dense_activation']),
            tf.keras.layers.Dense(1),
        ])

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss='mean_squared_error',
                                 optimizer=opt,
                                 metrics=["mse", "mae"])

        self.train_model.load_weights(p['weight_file'])

        return self.train_model

    def talos_model(self, X_train, y_train, x_val, y_val, p):
        input_shape = X_train.shape[1:]
        
        self.train_model = Sequential([
            tf.keras.layers.Conv1D(filters=p['first_conv_dim'], kernel_size=p['first_conv_kernel'],
                                   strides=1, padding="causal",
                                   activation=p['first_conv_activation'],
                                   input_shape=input_shape),
            tf.keras.layers.LSTM(p['second_lstm_dim']),
            tf.keras.layers.Dense(p['first_dense_dim'], activation=p['first_dense_activation']),
            tf.keras.layers.Dense(1),
        ])

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss='mean_squared_error',
                                 optimizer=opt,
                                 metrics=["mse", "mae"])
        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        return history, self.model

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.train_model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf',
                              save_format="tf")
        self.count_save += 1
