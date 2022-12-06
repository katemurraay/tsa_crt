"""
Random Forest (RF) predictive model
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from models.model_interface import ModelInterface
import time

class RF(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.p = {'n_estimators': 100,
                  'criterion': "squared_error",
                  'max_depth': None,
                  'max_features': "sqrt",
                  'bootstrap': True,
                  }
        """dict: Dictionary of Hyperparameter configuration of the model"""
        self.parameter_list = {'n_estimators': [150, 180, 200, 250, 300, 350, 400, 500],
                                        'criterion': ["squared_error", "absolute_error", "poisson"],
                                        'max_depth': [None, 10, 20, 50, 70, 100],
                                        'max_features': ["auto", "sqrt" ,"log2"],
                                        'bootstrap': [True, False],
                                       
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
        self.model = RandomForestRegressor(n_estimators=self.p['n_estimators'],
                                           criterion=self.p['criterion'],
                                           max_depth=self.p['max_depth'],
                                           max_features=self.p['max_features'],
                                           bootstrap=self.p['bootstrap'],)

    def fit(self):
        """
        Training of the model
        :return: None
        """
        self.__history_X = self.ds.X_train[:, :, 0]
        self.__history_y = np.ravel(self.ds.y_train.reshape(-1, 1))

        st = time.time()
        self.model = self.model.fit(self.__history_X, self.__history_y)  # .ravel())
        et = time.time()
        self.train_time = et-st
        return self.model

    def predict(self, X, train = False):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array: predictions: Predictions of the samples X
        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return

        X = X[:, :, 0]

        st = time.time()
        predictions = self.model.predict(X)
        et = time.time()
        self.inference_time = ((et-st) *1000)/len(X)
        if train: true = self.ds.y_train_array
        else: true = self.ds.y_test_array
        mse = mean_squared_error(true[:len(predictions)], predictions)
        mae = mean_absolute_error(true[:len(predictions)], predictions)

        if self.verbose:
            print('MSE: %.3f' % mse)
            print('MAE: %.3f' % mae)

        return predictions

    def hyperparametrization(self):
        """
        Search the best parameter configuration
        :return: None
        """
        self.__history_X = self.ds.X_train[:, :, 0]
        self.__history_y = np.ravel(self.ds.y_train.reshape(-1, 1))
        print('ML X_train shape: ',self.__history_X.shape)
        print('ML Y_train shape: ', self.__history_y.shape)
        self.__temp_model = RandomForestRegressor()
        split_val = int(len(self.__history_X) * 0.8) 
        tscv = TimeSeriesSplit(gap = 0, n_splits= 10, max_train_size=split_val)
        mse_score = make_scorer(mean_squared_error, greater_is_better=False)

        rf_gs = GridSearchCV(estimator=self.__temp_model, cv=tscv, param_grid=self.parameter_list, scoring=mse_score,
                             verbose=self.verbose)
        rf_gs.fit(self.__history_X, self.__history_y)
        df = pd.DataFrame(rf_gs.cv_results_)
        df.to_csv("talos/" + self.name + ".csv")

        print("BEST MODEL", rf_gs.best_estimator_)
        print("BEST PARAMS", rf_gs.best_params_)
        print("BEST SCORE", rf_gs.best_score_)

        self.p['n_estimators'] = rf_gs.best_params_['n_estimators']
        self.p['criterion'] = rf_gs.best_params_['criterion']
        self.p['max_depth'] = rf_gs.best_params_['max_depth']
        self.p['max_features'] = rf_gs.best_params_['max_features']
        self.p['bootstrap'] = rf_gs.best_params_['bootstrap']
  