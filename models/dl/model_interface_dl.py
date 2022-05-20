"""
Interface of a predictive DL model with shared functionalities
Inherits from ModelInterface class
"""

from models.model_interface import ModelInterface
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from util import custom_keras
import tensorflow as tf
import talos


class ModelInterfaceDL(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super(ModelInterfaceDL, self).__init__(name)

        self.count_save = 0
        """int: counter of the model saving"""
        self.best_val_loss = np.Inf
        """float: best validation found so far"""

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
        return history, self.model

    def predict(self, X):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array: predictions: Predictions of the samples X
        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        predictions = self.model.predict(X)
        return predictions

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

    def evaluate(self):
        """
        Evaluate the model on the training set ds.X_train
        :return: np.array: predictions: predictions of the trained model on the ds.X_train set
        """
        return self.predict(self.ds.X_train)

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

    def save_model(self):
        """
        Save the model into a file in self.saved_models directory
        :return: boolean: 1 if saving operating is successful, 0 otherwise
        """
        if self.model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf',
                        save_format="tf")
        self.count_save += 1
        return 1

    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        self.model.load(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf',
                        save_format="tf")
        self.count_save += 1
        return 1
