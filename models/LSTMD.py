import numpy as np
import tensorflow as tf
import talos
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Activation, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_probability as tfp
from util import custom_keras
from models.model_interface import ModelInterface
from sklearn.metrics import mean_squared_error
from util import plot_training
from keras.utils.vis_utils import plot_model
from datetime import datetime


class LSTMDPredictor(ModelInterface):
    def __init__(self):
        ModelInterface.__init__(self, "LSTMDPredictor")

        self.input_shape = None
        self.train_model = None
        self.model = None
        self.parameter_list = {'first_conv_dim': [32, 64, 128],
                               'first_conv_kernel': [3, 5, 7, 11],
                               'first_conv_activation': ['relu', 'tanh'],
                               'second_lstm_dim': [16, 32, 64],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': ['relu', 'tanh'],
                               'dense_kernel_init': ['he_normal', 'glorot_uniform'],
                               'batch_size': [256, 512, 1024],
                               'epochs': [2000],
                               'patience': [50],
                               'optimizer': ['adam', 'rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               'momentum': [0.9],
                               'decay': [1E-3, 1E-4, 1E-5],
                               }

    def compute_predictions(self, model, X_test):
        prediction_distribution = model(X_test)
        prediction_mean = prediction_distribution.mean().numpy().tolist()
        prediction_stdv = prediction_distribution.stddev().numpy().tolist()

        return prediction_mean, prediction_stdv

    def training(self, X_train, y_train, X_test, y_test, p):
        training_start = datetime.now()
        history, self.model = self.talos_model(X_train, y_train, X_test, y_test, p)
        training_time = datetime.now() - training_start

        inference_start = datetime.now()
        prediction_mean, prediction_std = self.compute_predictions(self.model, X_test)
        inference_time = (datetime.now() - inference_start) / y_test.shape[0]

        prediction_mean = np.concatenate(prediction_mean, axis=0)
        prediction_std = np.concatenate(prediction_std, axis=0)

        return self.model, history, prediction_mean, prediction_std, training_time, inference_time

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

    def talos_model(self, X_train, y_train, x_val, y_val, p):
        tf.keras.backend.clear_session()
        input_shape = X_train.shape[1:]

        input_tensor = Input(shape=input_shape)

        x = Conv1D(filters=p['first_conv_dim'], kernel_size=p['first_conv_kernel'],
                   strides=1, padding="causal",
                   activation=p['first_conv_activation'],
                   input_shape=input_shape)(input_tensor)

        x = LSTM(p['second_lstm_dim'])(x)

        x = layers.Dense(p['first_dense_dim'], activation=p['first_dense_activation'])(x)

        distribution_params = layers.Dense(units=2)(x)
        outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        self.train_model = Model(inputs=input_tensor, outputs=outputs)

        if p['optimizer'] == 'adam':
            opt = Adam(learning_rate=p['lr'], decay=p['decay'])
        elif p['optimizer'] == 'rmsprop':
            opt = RMSprop(learning_rate=p['lr'])
        elif p['optimizer'] == 'nadam':
            opt = Nadam(learning_rate=p['lr'])
        elif p['optimizer'] == 'sgd':
            opt = SGD(learning_rate=p['lr'], momentum=p['momentum'])
        self.train_model.compile(loss=self.negative_loglikelihood,
                                 optimizer=opt,
                                 metrics=["mse", "mae"])

        save_check = custom_keras.CustomSaveCheckpoint(self)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=p['patience'])

        history = self.train_model.fit(X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'],
                                       validation_split=0.2, verbose=2, callbacks=[es, save_check])

        self.model = save_check.dnn.model

        return history, self.model

    def negative_loglikelihood(self, targets, estimated_distribution):
        return -estimated_distribution.log_prob(targets)

    def prior(self, kernel_size, bias_size, dtype=None):
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
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n)),
                tfp.layers.MultivariateNormalTriL(n),
            ]
        )
        return posterior_model

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return

        self.train_model.save_weights(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.h5',
                                      save_format="h5")
        self.count_save += 1
