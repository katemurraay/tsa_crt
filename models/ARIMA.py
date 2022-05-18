from arch.__future__ import reindexing
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from models.model_interface import ModelInterface


class ARIMAPredictor:
    def __init__(self):
        ModelInterface.__init__(self, "ARIMAPredictor")
        self.train_model = None
        self.model = None
        self.parameter_list = {'p': 2,
                               'd': 0,
                               'q': 2,
                               'P': 0,
                               'Q': 0,
                               'D': 0,
                               'S': 12,
                               'selection': False,
                               'loop': 0,
                               'horizon': 0,
                               'sliding_window': 288
                               }
        self.history = None

    def training(self, X_train, y_train, X_test, y_test, p):
        X_train = list(X_train)

        self.history = X_train
        if p is not None:
            self.parameter_list = p
        if self.parameter_list['sliding_window']:
            self.history = self.history[-self.parameter_list['sliding_window']:]
        if self.parameter_list['selection']:
            self.param_selection(X_train)
        self.model = ARIMA(X_train, order=(self.parameter_list['p'], self.parameter_list['d'],
                                           self.parameter_list['q']),
                           seasonal_order=(self.parameter_list['P'], self.parameter_list['D'],
                                           self.parameter_list['Q'], self.parameter_list['S']))
        self.train_model = self.model.fit(method_kwargs={"warn_convergence": False})

        print(self.train_model.summary())
        print(self.train_model.params)

        predicted_mean, predicted_std, _ = self.predict(X_test, self.parameter_list['loop'],
                                                        self.parameter_list['horizon'])

        return predicted_mean, predicted_std, self.train_model

    def predict(self, X, steps, horizon):
        if self.train_model is None:
            print("ERROR: the model needs to be trained before predict")
            return

        predicted_means, predicted_stds = list(), list()
        if steps == 0 and not self.parameter_list['loop']:
            steps = X.shape[0]
        elif steps == 0 and self.parameter_list['loop']:
            steps = 1

        for j in range(X.shape[0] // steps):
            t = j * steps
            self.model = ARIMA(self.history, order=(self.parameter_list['p'], self.parameter_list['d'],
                                                    self.parameter_list['q']),
                               seasonal_order=(self.parameter_list['P'], self.parameter_list['Q'],
                                               self.parameter_list['D'], self.parameter_list['S']))

            # retrain the model at each step prediction
            self.train_model = self.model.fit(method_kwargs={"warn_convergence": False})
            result = self.train_model.get_forecast(steps=int(steps + horizon))  # steps=steps + horizon)

            predicted_mean = result.predicted_mean
            predicted_std = result.se_mean

            yhat = predicted_mean[horizon:]
            [predicted_means.append(em) for em in predicted_mean[horizon:]]
            [predicted_stds.append(em) for em in predicted_std[horizon:]]
            obs = X[j * steps + horizon:(j + 1) * steps + horizon]
            [self.history.append(a) for a in X[j * steps:(j + 1) * steps]]

            if self.parameter_list['sliding_window']:
                self.history = self.history[-self.parameter_list['sliding_window']:]
            print(t, " predicted = ", yhat, ' expected ', obs)

        # evaluate forecasts
        X = np.concatenate(X, axis=0)

        rmse = sqrt(mean_squared_error(X[:len(predicted_means)], predicted_means))
        print('Test RMSE: %.3f' % rmse)
        self.history = list(self.history)
        return predicted_means, predicted_stds, self.history

    # evaluate an GARCH model for a given order (p,q)
    def evaluate_arima_model(self, X, arima_order, arima_seasonal_order):
        # prepare training dataset
        train_size = int(len(X) * 0.9)
        train, test = X[0:train_size], X[train_size:]

        history = [x for x in train]

        predicted_means, predicted_stds = list(), list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order,
                          seasonal_order=arima_seasonal_order,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            model_fit = model.fit(method_kwargs={"warn_convergence": False})
            result = model_fit.get_forecast()
            yhat = result.predicted_mean
            predicted_means.append(yhat)
            predicted_stds.append(result.se_mean)
            history.append(test[t])
            
        # calculate out of sample error
        error = mean_squared_error(test, predicted_means)
        return np.sqrt(error)

    def param_selection(self, X_train):
        p = q = range(0, 3)
        d = [0]
        pdq = list(itertools.product(p, d, q))

        pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        ans = []
        for comb in pdq:
            for combs in pdqs:
                try:
                    mod = ARIMA(X_train,
                                order=comb,
                                seasonal_order=combs,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                    output = mod.fit(method_kwargs={"warn_convergence": False})
                    rmse = self.evaluate_arima_model(X_train, comb, combs)
                    ans.append([comb, combs, output.aic, rmse])
                    print('Arima {} x {} : AIC Calculated = {}, RMSE Calculated = {}'.format(comb, combs, output.aic,
                                                                                             rmse))
                except:
                    continue

        ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic', 'rmse'])
        print(ans_df)
        best = ans_df.loc[ans_df['rmse'].idxmin()]
        self.parameter_list['p'], self.parameter_list['d'], self.parameter_list['q'] = best['pdq']
        self.parameter_list['P'], self.parameter_list['Q'], self.parameter_list['D'], self.parameter_list['S'] = \
            best['pdqs']

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.train_model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.pkl')
        self.count_save += 1

    def load_model(self, name):
        self.model = ARIMAResults.load(self.model_path + name + '_model.pkl')
