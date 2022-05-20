"""
Interface of a predictive probabilistic model with shared functionalities
Inherits from ModelInterface and ModelPorbabilistic class
"""
from models.model_probabilistic import ModelProbabilistic
from models.dl.model_interface_dl import ModelInterfaceDL

import numpy as np


class ModelProbabilisticDL(ModelProbabilistic, ModelInterfaceDL):
    def __init__(self, name):
        """
        Constructor of the Model Probabilistic class
        :param name: string: name of the model
        """
        # super(ModelProbabilistic, self).__init__(name)
        super(ModelProbabilisticDL, self).__init__(name)

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
        """
        if self.ds is None:
            print("ERROR: dataset not linked")
        self.fit()
        prediction_mean, prediction_std = self.predict(X)
        return prediction_mean, prediction_std

    def evaluate(self):
        """
        Evaluate the model on the training set ds.X_train_array
        :return: np.array: prediction_mean: predictions of the mean of the samples X
                 np.array: prediction_std: predictions of the standard deviation of the samples X
        """
        if self.verbose:
            print("Evaluate")
        predicted_mean, predicted_std = self.predict(self.ds.X_train)
        return predicted_mean, predicted_std

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


