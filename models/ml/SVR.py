"""
Interface of a predictive model with shared functionalities
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from joblib import load, dump
from models.model_interface import ModelInterface
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from models.model_interface import ModelInterface


class SVR(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.p = {'kernel': 'rbf',
                  'degree': 3,
                  'tol': 0.001,
                  'C': 1.0,
                  'sliding_window': 288
                  }
        self.parameter_list = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                               'degree': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                               }

        self.tuning_window = 0
        """int:"""

        self.history_X = None
        """np.array: temporary training features"""
        self.history_y = None
        """np.array: temporary training labels"""

        # Model configuration
        self.verbose = True
        """Boolean: Print output of the training phase"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        self.model = svm.SVR(kernel=self.p['kernel'],
                             degree=self.p['degree'])

    def fit(self):
        """
        Training of the model
        :return: None
        """
        self.history_X = self.ds.X_train
        self.history_y = self.ds.y_train

        if self.p['sliding_window'] > 0:
            self.history_X = self.ds.X_train[-self.p['sliding_window']:]
            self.history_y = self.ds.y_train[-self.p['sliding_window']:]

        self.model = self.model.fit(self.history_X, self.history_y.ravel())

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

        self.history_X = self.ds.X_train
        self.history_y = self.ds.y_train

        predicted_mean = list()
        steps = X.shape[0]

        # manage also the case when X.shape[0] % steps !=0
        iterations = X.shape[0] // steps + 1 * (X.shape[0] % steps != 0)
        for j in range(iterations):
            # last iterations predict over the last remaining steps

            self.temp_model = svm.SVR(kernel=self.p['kernel'],
                                      degree=self.p['degree']
                                      )

            self.model = self.temp_model.fit(self.history_X, self.history_y)
            print(j, steps, j * steps, (j + 1) * steps)
            output = self.model.predict(X[j * steps:(j + 1) * steps])

            [predicted_mean.append(em) for em in output]

        mse = mean_squared_error(X[:len(predicted_mean)], predicted_mean)
        mae = mean_absolute_error(X[:len(predicted_mean)], predicted_mean)

        if self.verbose:
            print('MSE: %.3f' % mse)
            print('MAE: %.3f' % mae)
        return predicted_mean

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
        if self.verbose:
            print("Evaluate")
        return self.predict(self.ds.X_train)

    def save_model(self):
        """
        Save the model into a file in self.saved_models directory
        :return: boolean: 1 if saving operating is successful, 0 otherwise
        """
        if self.model is None:
            print("ERROR: the model must be available before saving it")
            return
        dump(self.model, self.model_path + self.name + '_model.joblib')
        return 1

    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        self.model = load(self.model_path + self.name + '_model.joblib')
        return 1

    def hyperparametrization(self):
        """
        Search the best parameter configuration
        :return: None
        """
        self.temp_model = svm.SVR()
        tscv = TimeSeriesSplit(n_splits=10)
        mse_score = make_scorer(mean_squared_error, greater_is_better=False)

        svr_gs = GridSearchCV(estimator=self.temp_model, cv=tscv, param_grid=self.parameter_list, scoring=mse_score)
        svr_gs.fit(self.ds.X_train, self.ds.y_train.ravel())

        print("BEST MODEL", svr_gs.best_estimator_)
        print("BEST PARAMS", svr_gs.best_params_)
        print("BEST SCORE", svr_gs.best_score_)

        self.p['kernel'] = svr_gs.best_params_['kernel']
        self.p['degree'] = svr_gs.best_params_['degree']
