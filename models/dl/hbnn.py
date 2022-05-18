"""
Hybrid Bayesian Neural Network model
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


class HBNN(ModelProbabilisticDL):
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
                               'first_dense_activation': [keras.activations.relu, keras.activations.tanh],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2],
                               'patience': [50],
                               'optimizer': ['adam', 'rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'momentum': [0.9, 0.99],
                               'decay': [1E-3, 1E-4, 1E-5],
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

        x = Conv1D(filters=self.p['first_conv_dim'], kernel_size=5,
                   strides=1, padding="causal",
                   activation=self.p['first_conv_activation'],
                   input_shape=input_shape)(input_tensor)

        x = LSTM(self.p['first_lstm_dim'])(x)

        x = self.VarLayer('var', self.p['first_dense_dim'],
                          self.prior,
                          self.posterior,
                          1 / self.ds.X_train.shape[0],
                          self.p['first_dense_activation'])(x)
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
        self.temp_model.compile(loss=self.negative_loglikelihood,
                                optimizer=opt,
                                metrics=["mse", "mae"])

    def fit(self):
        """
        Training of the model
        :return: None
        """
        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.p['patience'])

        history = self.temp_model.fit(self.ds.X_train, self.ds.y_train, epochs=self.p['epochs'],
                                      batch_size=self.p['batch_size'],
                                      validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

    def tune(self, X, y):
        """
        Tune the models with new available samples (X, y)
        :param X: nparray: Training features
        :param y: nparray: Training labels
        :return: None
        """
        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.p['patience'])

        history = self.temp_model.fit(X, y, epochs=self.p['epochs'], batch_size=self.p['batch_size'],
                                      validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

    def predict(self, X):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array: prediction_mean: predictions of the mean of the samples X
                 np.array: prediction_std: predictions of the standard deviation of the samples X
        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        predictions = self.model(X)
        prediction_mean = predictions.mean().numpy()
        prediction_std = predictions.stddev().numpy()
        return prediction_mean, prediction_std

    def fit_predict(self, X):
        """
        Training the model on self.ds.X_train and self.ds.y_train and predict on samples X
        :param X: np.array: Input samples to predict
        :return: np.array: prediction_mean: predictions of the mean of the samples X
                 np.array: prediction_std: predictions of the standard deviation of the samples X
        """
        if self.ds is None:
            print("ERROR: dataset not linked")
        self.fit()
        prediction_mean, prediction_std = self.predict(X)
        return prediction_mean, prediction_std

    def evaluate(self):
        """
        Evaluate the model on the training set ds.X_train
        :return: np.array: predictions: predictions of the trained model on the ds.X_train set
        """
        return self.predict(self.ds.X_train)

    def save_model(self):
        """
        Save the model into a file in self.saved_models directory
        :return: boolean: 1 if saving operating is successful, 0 otherwise
        """
        if self.temp_model is None:
            print("ERROR: the model must be available before saving it")
            return

        self.temp_model.save_weights(self.model_path + self.name + str(self.count_save).zfill(4) + '_weights.tf',
                                     save_format="tf")

        self.count_save += 1
        return 1

    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        self.create_model()
        self.model.load_weights(self.p['weight_file'])
        return 1

    def hyperparametrization(self):
        """
        Search the best parameter configuration using talos
        :return: None
        """
        tf.keras.backend.clear_session()
        talos.Scan(x=self.ds.X_train,
                   y=self.ds.y_train,
                   model=self.__talos_model,
                   experiment_name='talos/' + self.name,
                   params=self.parameter_list,
                   clear_session=True,
                   print_params=True,
                   round_limit=5)

    def __talos_model(self, X_train, y_train, x_val, y_val, p):
        """
        Custom fuction for talos optimization
        :return: fit function
        """
        self.create_model()
        return self.fit()

    def prior(self, kernel_size, bias_size, dtype=None):
        """
        Prior probability distribution function
        :param kernel_size: int: kernel size
        :param bias_size: int: bias size
        :param dtype: data type
        :return: keras model as a multivariate normal distribution of the specified size
        """
        n = kernel_size + bias_size
        prior_model = keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model

    def posterior(self, kernel_size, bias_size, dtype=None):
        """
        Posterior probability distribution function
        :param kernel_size: int: kernel size
        :param bias_size: int: bias size
        :param dtype: data type
        :return: keras model as a multivariate normal distribution of the specified size
        """
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n)),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

    def negative_loglikelihood(self, targets, estimated_distribution):
        """
        Negative log-likelihood custom function
        :param targets: np.array: labels
        :param estimated_distribution: np.array: prediction
        :return:
        """
        return -estimated_distribution.log_prob(targets)

    @tf.keras.utils.register_keras_serializable()
    class VarLayer(tfp.layers.DenseVariational):
        """
        Variational Dense Layer inherits from tfp.layers.DenseVariational
        """

        def __init__(self, name, units, make_prior_fn, make_posterior_fn, kl_weight, activation, **kwargs):
            """
            Constructor of the variational dense layer
            :param name: string: name of the layer
            :param units: int: number of neurons
            :param make_prior_fn: func: priori probability function
            :param make_posterior_fn: func: posteriori probability function
            :param kl_weight: float: kl weight
            :param activation: keras.activation.function: activation function of the layer
            :param kwargs: dict: extra arguments (see tfp.layers.DenseVariational documentation)
            """
            super().__init__(units=units, make_prior_fn=make_prior_fn, make_posterior_fn=make_posterior_fn, name=name,
                             kl_weight=kl_weight, activation=activation, **kwargs)

        def get_config(self):
            """
            configuration of the layer
            :return: None
            """
            config = super(HBNN.VarLayer, self).get_config()
            config.update({
                'name': self.name,
                'units': self.units,
                'activation': self.activation})
            return config

        def call(self, inputs):
            """
            Method necessary for talos implementation to make this class callable
            :param inputs: parameters
            :return:
            """
            return super(HBNN.VarLayer, self).call(inputs)
