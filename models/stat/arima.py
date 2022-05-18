"""
ARIMA model
Inherits from ModelInterface class and build over statsmodel.tsa.arima
"""
import pickle

from models.model_probabilistic import ModelProbabilistic
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pandas as pd
import statsmodels.tsa.arima as arima
from statsmodels.tsa.arima_model import ARIMAResults


class ARIMA(ModelProbabilistic):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.parameter_list = {}
        """dict: Dictionary of hyperparameters search space"""
        self.p = {'p': 2,
                  'd': 0,
                  'q': 2,
                  'P': 0,
                  'Q': 0,
                  'D': 0,
                  'S': 12,
                  'loop': 0,
                  'horizon': 0,
                  'sliding_window': 288
                  }
        """dict: Dictionary of Hyperparameter configuration of the model"""
        self.history = None
        """np.array: temporary training set"""

        # Model configuration
        self.verbose = True
        """Boolean: Print output of the training phase"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        self.model = arima.model.ARIMA(self.ds.X_train, order=(self.p['p'], self.p['d'],
                                                               self.p['q']),
                                       seasonal_order=(self.p['P'], self.p['Q'],
                                                       self.p['D'], self.p['S']))

    def fit(self):
        """
        Training of the model
        :return: None
        """
        self.history = list(self.ds.X_train)
        if self.p['sliding_window']:
            self.history = self.history[-self.p['sliding_window']:]
        self.model = arima.model.ARIMA(self.ds.X_train, order=(self.p['p'], self.p['d'],
                                                               self.p['q']),
                                       seasonal_order=(self.p['P'], self.p['D'],
                                                       self.p['Q'], self.p['S']))
        self.temp_model = self.model.fit(method_kwargs={"warn_convergence": False})

        if self.verbose:
            print(self.temp_model.summary())
            print(self.temp_model.params)

    def tune(self, X, y):
        """
        Tune the models with new available samples (X, y)
        :param X: nparray: Training features
        :param y: nparray: Training labels
        :return: None
        """
        self.history = X
        self.model = arima.model.ARIMA(X, order=(self.p['p'], self.p['d'],
                                                 self.p['q']),
                                       seasonal_order=(self.p['P'], self.p['D'],
                                                       self.p['Q'], self.p['S']))

        self.model.fit(method_kwargs={"warn_convergence": False})

        if self.verbose:
            print(self.model.summary())
            print("Parameters", self.model.params)

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
        self.history = list(self.ds.X_train)
        predicted_means, predicted_stds = list(), list()
        if self.p['loop'] == 0:
            steps = X.shape[0]
        else:
            steps = self.p['loop']

        # manage also the case when X.shape[0] % steps !=0
        iterations = X.shape[0] // steps + 1 * (X.shape[0] % steps != 0)
        for j in range(iterations):
            # last iterations predict over the last remaining steps
            if j == iterations - 1:
                steps = X.shape[0] % steps
            self.model = arima.model.ARIMA(self.history, order=(self.p['p'], self.p['d'],
                                                                self.p['q']),
                                           seasonal_order=(self.p['P'], self.p['Q'],
                                                           self.p['D'], self.p['S']))

            # retrain the model at each step prediction
            self.temp_model = self.model.fit(method_kwargs={"warn_convergence": False})
            result = self.temp_model.get_forecast(steps=int(steps + self.p['horizon']))

            predicted_mean = result.predicted_mean
            predicted_std = result.se_mean

            yhat = predicted_mean[self.p['horizon']:]
            [predicted_means.append(em) for em in predicted_mean[self.p['horizon']:]]
            [predicted_stds.append(em) for em in predicted_std[self.p['horizon']:]]
            obs = X[j * steps + self.p['horizon']:(j + 1) * steps + self.p['horizon']]
            [self.history.append(a) for a in X[j * steps:(j + 1) * steps]]

            if self.p['sliding_window']:
                self.history = self.history[-self.p['sliding_window']:]

        mse = mean_squared_error(X[:len(predicted_means)], predicted_means)
        mae = mean_absolute_error(X[:len(predicted_means)], predicted_means)

        if self.verbose:
            print('MSE: %.3f' % mse)
            print('MAE: %.3f' % mae)
        self.history = list(self.history)
        return predicted_means, predicted_stds

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
        Evaluate the model on the training set ds.X_train
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
        if self.model is None:
            print("ERROR: the model must be available before saving it")
            return
        pickle.dump(self.model, self.model_path + self.name + '_model.pkl')
        return 1

    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        self.model = pickle.load(self.model_path + self.name + '_model.pkl')

    def hyperparametrization(self):
        """
        Search the best parameter configuration
        :return: None
        """
        p = q = range(0, 3)
        d = [0]
        pdq = list(itertools.product(p, d, q))

        pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        ans = []
        for comb in pdq:
            for combs in pdqs:
                # try:
                mod = arima.model.ARIMA(self.ds.X_train,
                                        order=comb,
                                        seasonal_order=combs,
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                output = mod.fit(method_kwargs={"warn_convergence": False})
                rmse = self.__evaluate_arima_model(self.ds.X_train, comb, combs)
                ans.append([comb, combs, output.aic, rmse])
                print('Arima {} x {} : AIC Calculated = {}, RMSE Calculated = {}'.format(comb, combs, output.aic,
                                                                                         rmse))
                # except:
                #     continue

        ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic', 'rmse'])
        if self.verbose:
            print(ans_df)

        print(ans_df)

        best = ans_df.loc[ans_df['rmse'].idxmin()]
        self.p['p'], self.p['d'], self.p['q'] = best['pdq']
        self.p['P'], self.p['Q'], self.p['D'], self.p['S'] = \
            best['pdqs']
        ans_df.to_csv("talos/" + self.name + ".csv")

    def __evaluate_arima_model(self, X, arima_order, arima_seasonal_order):
        """
        Validation of the hyperparameter
        :param X: Training features
        :param arima_order: p,d,q order
        :param arima_seasonal_order: P,D,Q,S seasonal order
        :return: RMSE of the validation set
        """
        # prepare training dataset
        train_size = int(len(X) * 0.9)
        train, test = X[0:train_size], X[train_size:]

        history = [x for x in train]

        predicted_means, predicted_stds = list(), list()
        for t in range(len(test)):
            if self.verbose:
                if t % 100 == 0:
                    print("Iteration ", t, ' of ', len(test))
            self.model = arima.model.ARIMA(history, order=arima_order,
                                           seasonal_order=arima_seasonal_order,
                                           enforce_stationarity=False,
                                           enforce_invertibility=False)
            self.temp_model = self.model.fit(method_kwargs={"warn_convergence": False})
            result = self.temp_model.get_forecast()
            yhat = result.predicted_mean
            predicted_means.append(yhat)
            predicted_stds.append(result.se_mean)
            history.append(test[t])

        # calculate out of sample error
        error = mean_squared_error(test, predicted_means)
        return np.sqrt(error)
