from datetime import datetime

"""
Interface of a predictive model with shared functionalities
"""
class ModelInterface:
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        self.name = name
        """string: Name of the Model"""
        self.model = None
        """Model: The best trained model"""
        self.temp_model = None
        """Model: Temporary model in the training phase"""
        self.ds = None
        """Dataset: Dataset used for training the model"""
        self.model_path = './saved_models/'
        """string: Path of the dictory where to save the model"""
        self.parameter_list = {}
        """dict: Dictionary of hyperparameters search space"""
        self.p = {}
        """dict: Dictionary of Hyperparameter configuration of the model"""

        # Model configuration
        self.verbose = False
        """Boolean: Print output of the training phase"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        pass

    def fit(self):
        """
        Training of the model
        :return: None
        """
        pass

    def tune(self, X, y):
        """
        Tune the models with new available samples (X, y)
        :param X: nparray: Training features
        :param y: nparray: Training labels
        :return: None
        """
        pass

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

    def fit_predict(self, X):
        """
        Training the model on self.ds.X_train and self.ds.y_train and predict on samples X
        :param X: np.array: Input samples to predict
        :return: np.array: predictions: predictions of the samples X
        """
        if self.ds is None:
            print("ERROR: dataset not linked")
        self.fit()
        predictions = self.predict(X)
        return predictions

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
        return 0

    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        pass

    def hyperparametrization(self):
        """
        Search the best parameter configuration
        :return: None
        """
        pass
