"""
Interface of a predictive probabilistic model with shared functionalities
Inherits from ModelInterface class
"""

from model_interface import ModelInterface


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
