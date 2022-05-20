"""
Interface of a predictive probabilistic model with shared functionalities
Inherits from ModelInterface class
"""

from models.model_interface import ModelInterface


class ModelProbabilistic(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Probabilistic class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.is_probabilistic = True
        """Boolean: Determines if the model is probabilistic"""

    def predict(self, X):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array or tfp.distributions.Distribution: Probability distribution of the prediction of the samples X
        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        predictions = self.model.predict(X)
        return predictions

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
        predicted_means, predicted_stds = self.predict(X)
        return predicted_means, predicted_stds

    def evaluate(self):
        """
        Evaluate the model on the training set ds.X_train_array
        :return: np.array: prediction_mean: predictions of the mean of the samples X
                 np.array: prediction_std: predictions of the standard deviation of the samples X
        """
        if self.verbose:
            print("Evaluate")
        predicted_mean, predicted_std = self.predict(self.ds.X_train_array)
        return predicted_mean, predicted_std
