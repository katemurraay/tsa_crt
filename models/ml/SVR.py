from datetime import datetime

import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from joblib import load, dump
from models.model_interface import ModelInterface
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit


class SVRPredictor(ModelInterface):
    def __init__(self):
        ModelInterface.__init__(self, "SVRPredictor")
        self.p = {'kernel': 'rbf',
                               'degree': 3,
                               'tol': 0.001,
                               'C': 1.0,
                               'sliding_window': 288
                               }
        self.parameter_list = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                      'degree': [2,3,4,5,6,7,8,9,10],
                      #'tol': [1e-3, 1e-4, 1e-5],
                      #'C': [1, 2]
                      }

        self.tuning_window = 0

    # Initialize the model
    def create_model(self):
        self.model = SVR(kernel=self.p['kernel'],
                         degree=self.p['degree'])

    def fit(self):
        start_time = datetime.now()
        if self.parameter_list['sliding_window']>0:
            X_train = self.ds.X_train[-self.parameter_list['sliding_window']:]
            y_train = self.ds.y_train[-self.parameter_list['sliding_window']:]

        self.model = self.model.fit(X_train, y_train.ravel())
        return datetime.now() - start_time


    def predict(self, X):
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return

        predicted_mean = list()
        X = self.ds.X_timeseries
        X_len = X.shape[0]
        # predicted_stdv = list()
        if self.tuning_window == 0:
            self.tuning_window = X_len
        for j in range(X_len // self.tuning_window):
            start = j * self.tuning_window

            pred = self.model.predict(X[j * self.tuning_window:(j + 1) * self.tuning_window])  # horizon=steps + horizon, reindex=False)

            # yhat_mean = output.mean.iloc[-1]
            # yhat_var = output.variance.iloc[-1]

            [predicted_mean.append(em) for em in pred]
            obs = X[j * self.tuning_window + self.ds.horizon:(j + 1) * self.tuning_window + self.ds.horizon]
            self.history = list(self.history[0]), list(self.history[1])
            [self.history[0].append(a) for a in X[j * self.tuning_window:(j + 1) * self.tuning_window]]
            [self.history[1].append(a) for a in y[j * self.tuning_window:(j + 1) * self.tuning_window]]

            # self.history = pd.DataFrame(self.history)
            if self.parameter_list['sliding_window']:
                self.history = self.history[0][-self.parameter_list['sliding_window']:], \
                               self.history[1][-self.parameter_list['sliding_window']:]

            # print(t, " predicted mean = ", yhat_mean, " predicted variance = ", yhat_var, ' expected ', obs)

        # evaluate forecasts
        X = np.concatenate(X, axis=0)

        rmse = sqrt(mean_squared_error(X[:len(predicted_mean)], predicted_mean))
        print('Test RMSE: %.3f' % rmse)
        # self.history = list(self.history.values)
        return predicted_mean, self.history  # .train_model.params, self.history

    def param_selection(self, history):
        X_train, y_train = history

        parameters = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                      'degree': [2,3,4,5,6,7,8,9,10],
                      #'tol': [1e-3, 1e-4, 1e-5],
                      #'C': [1, 2]
                      }

        mod = SVR()
        tscv = TimeSeriesSplit(n_splits=10)
        mse_score = make_scorer(self.mse, greater_is_better=False)

        svr_gs = GridSearchCV(estimator=mod, cv=tscv, param_grid=parameters, scoring=mse_score)
        svr_gs.fit(X_train, y_train.ravel())

        print("BEST MODEL", svr_gs.best_estimator_)
        print("BEST PARAMS", svr_gs.best_params_)
        print("BEST SCORE", svr_gs.best_score_)

        self.parameter_list['kernel'] = svr_gs.best_params_['kernel']
        self.parameter_list['degree'] = svr_gs.best_params_['degree']
        #self.parameter_list['tol'] = svr_gs.best_params_['tol']
        #self.parameter_list['C'] = svr_gs.best_params_['C']

    def regression_results(self, y_true, y_pred):
        # Regression metrics
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
        median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)
        print('explained_variance: ', round(explained_variance, 4))
        # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
        print('r2: ', round(r2, 4))
        print('MAE: ', round(mean_absolute_error, 4))
        print('MSE: ', round(mse, 4))
        print('RMSE: ', round(np.sqrt(mse), 4))

    def mse(self, actual, predict):
        predict = np.array(predict)
        actual = np.array(actual)
        score = mean_squared_error(actual, predict)
        # distance = predict - actual
        # square_distance = distance ** 2
        # mean_square_distance = square_distance.mean()
        # score = np.sqrt(mean_square_distance)
        return score

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return
        dump(self.train_model, self.model_path + self.name + str(self.count_save).zfill(4) + '_model.joblib')
        # self.train_model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.pkl')
        self.count_save += 1

    def load_model(self, name):
        load(name)
        return