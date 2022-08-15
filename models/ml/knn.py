"""
Suppor Vector Regressor (KNN) predictive model with shared functionalities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from models.model_interface import ModelInterface


class KNN(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.p = {'n_neighbors': 5,
                  'weights': 'distance',
                  'algorithm': 'auto',
                  'p': 2,
                  'sliding_window': 288
                  }
        """dict: Dictionary of Hyperparameter configuration of the model"""
        self.parameter_list = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40],
                               'weights': ('uniform', 'distance'),
                               'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                               'p': [1, 2]
                               }
        """dict: Dictionary of hyperparameters search space"""
        self.sliding_window = 0
        """int: sliding window to apply in the training phase"""

        self.__history_X = None
        """np.array: temporary training features"""
        self.__history_y = None
        """np.array: temporary training labels"""

        # Model configuration
        self.verbose = True
        """Boolean: Print output of the training phase"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        self.model = KNeighborsRegressor(n_neighbors=self.p['n_neighbors'],
                                         weights=self.p['weights'],
                                         algorithm=self.p['algorithm'],
                                         p=self.p['p'])

    def fit(self):
        """
        Training of the model
        :return: None
        """

        if self.sliding_window > 0:
            self.__history_X = self.ds.X_train_array[-self.sliding_window:]
            self.__history_y = self.ds.y_train_array[-self.sliding_window:]
        else: 
            self.__history_X = self.ds.X_train_array  # .reshape(-1, len(self.ds.training_features))
            self.__history_y = self.ds.y_train_array  # .reshape(-1, len(self.ds.target_name))

        self.model = self.model.fit(self.__history_X, self.__history_y)  # .ravel())

    def predict(self, X):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array: predictions: Predictions of the samples X
        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return

        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        if np.array_equal(X, self.ds.X_test):
            X = self.ds.X_test_array

        predictions = self.model.predict(X)
        mse = mean_squared_error(X[:len(predictions)], predictions)
        mae = mean_absolute_error(X[:len(predictions)], predictions)

        if self.verbose:
            print('MSE: %.3f' % mse)
            print('MAE: %.3f' % mae)

        return predictions

    def hyperparametrization(self):
        """
        Search the best parameter configuration
        :return: None
        """
        self.__temp_model = KNeighborsRegressor()
        tscv = TimeSeriesSplit(n_splits=10)
        mse_score = make_scorer(mean_squared_error, greater_is_better=False)

        knn_gs = GridSearchCV(estimator=self.__temp_model, cv=tscv, param_grid=self.parameter_list, scoring=mse_score)
        knn_gs.fit(self.ds.X_train_array, self.ds.y_train_array)

        print("BEST MODEL", knn_gs.best_estimator_)
        print("BEST PARAMS", knn_gs.best_params_)
        print("BEST SCORE", knn_gs.best_score_)

        self.p['n_neighbors'] = knn_gs.best_params_['n_neighbors']
        self.p['weights'] = knn_gs.best_params_['weights']
        self.p['algorithm'] = knn_gs.best_params_['algorithm']
        self.p['p'] = knn_gs.best_params_['p']
