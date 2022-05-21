"""
Interface of a predictive model with shared functionalities
"""
import pickle
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import pandas as pd
from arch import arch_model
from models.model_probabilistic import ModelProbabilistic


class GARCH(ModelProbabilistic):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.__temp_model = None
        self.parameter_list = {'p': 1,
                               'q': 1,
                               'loop': 1,
                               'horizon': 2,
                               'mean': ['Constant', 'LS', 'AR'],
                               'method': ['M-H', 'BBVI'],
                               }
        """dict: Dictionary of hyperparameters search space"""
        self.p = {'p': 1,
                  'q': 1,
                  'loop': 17*131,
                  'horizon': 0,
                  'mean': 'LS',
                  'sliding_window': 288
                  }
        """dict: Dictionary of Hyperparameter configuration of the model"""
        self.__history = None
        """np.array: temporary training set"""
        self.sliding_window = 288
        """int: sliding window to apply in the training phase"""

    def create_model(self):
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        self.model = arch_model(self.ds.X_train_array, mean=self.p['mean'], p=self.p['p'],
                                q=self.p['q'])

    def fit(self):
        """
        Training of the model
        :return: None
        """
        self.__history = pd.DataFrame(self.ds.X_train_array)
        if self.sliding_window:
            self.__history = self.__history.iloc[-self.sliding_window:]
        self.model = arch_model(self.__history, mean=self.p['mean'], p=self.p['p'],
                                q=self.p['q'])

        self.__temp_model = self.model.fit(disp='off', show_warning=False)

        if self.verbose:
            print(self.__temp_model.summary())

        # predictions, predicted_stdv = self.predict(self.ds.X_test)#, self.p['loop'],
        #  self.ds.horizon)

    def tune(self, X, y):
        """
        Tune the models with new available samples (X, y)
        :param X: nparray: Training features
        :param y: nparray: Training labels
        :return: None
        """
        self.__history = X
        self.model = arch_model(self.__history, mean=self.p['mean'], p=self.p['p'],
                                q=self.p['q'])

        self.model.fit(disp='off', show_warning=False)

        if self.verbose:
            print(self.model.summary())

    def predict(self, X):
        """
        Inference step on the samples X
        :param X: np.array: Input samples to predict
        :return: np.array: prediction_mean: predictions of the mean of the samples X
                 np.array: prediction_std: predictions of the standard deviation of the samples X        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        self.__history = pd.DataFrame(self.ds.X_train_array)
        if np.array_equal(X, self.ds.X_test):
            X = self.ds.X_test_array
        predicted_mean, predicted_std = list(), list()
        if self.p['loop'] == 0:
            steps = X.shape[0]
        else:
            steps = self.p['loop']

        # manage also the case when X.shape[0] % steps !=0
        iterations = X.shape[0] // steps + 1 * (X.shape[0] % steps != 0)
        for j in range(iterations):
            # last iterations predict over the last remaining steps
            if X.shape[0] % steps != 0 and j == iterations - 1:
                steps = X.shape[0] % steps
            self.model = arch_model(self.__history, mean=self.p['mean'], p=self.p['p'],
                                    q=self.p['q'])

            self.__temp_model = self.model.fit(disp='off',
                                               show_warning=False)
            # retrain the model at each step prediction
            output = self.__temp_model.forecast(horizon=steps + self.ds.horizon, reindex=False)
            [predicted_mean.append(em) for em in output.mean.iloc[-1]]
            [predicted_std.append(em) for em in np.sqrt(output.variance.iloc[-1])]
            self.__history = list(self.__history.values)
            [self.__history.append(a) for a in X[j * steps:(j + 1) * steps]]

            self.__history = pd.DataFrame(self.__history)
            if self.sliding_window:
                self.__history = self.__history.iloc[-self.sliding_window:]

        if self.verbose:
            mse = mean_squared_error(X, predicted_mean)
            mae = mean_absolute_error(X, predicted_mean)
            print('MSE: %.3f' % mse)
            print('MAE: %.3f' % mae)
        self.__history = list(self.__history.values)
        return predicted_mean, predicted_std

    def hyperparametrization(self):
        """
        Search the best parameter configuration
        :return: None
        """
        mean_mod = ['LS', 'Constant', 'AR']
        p = q = range(1, 3)
        pq = list(itertools.product(p, q))
        ans = []
        for mean in mean_mod:
            for comb in pq:
                mod = arch_model(self.ds.X_train_array,
                                 mean=mean,
                                 p=comb[0],
                                 q=comb[1])
                output = mod.fit(disp='off', show_warning=False)
                rmse = self.__evaluate_garch_model(self.ds.X_train_array, comb, mean)
                ans.append([comb, mean, output.aic, rmse])
                print('GARCH {} : Mean {} : AIC Calculated = {}, RMSE Calculated = {}'.format(comb, mean, output.aic,
                                                                                              rmse))
        ans_df = pd.DataFrame(ans, columns=['pq', 'mean', 'aic', 'rmse'])
        best = ans_df.loc[ans_df['rmse'].idxmin()]
        self.p['p'], self.p['q'] = best['pq']
        self.p['mean'] = best['mean']
        ans_df.to_csv("talos/" + self.name + ".csv")

    def __evaluate_garch_model(self, X, model_param, mean_model):
        # prepare training dataset
        train_size = int(X.shape[0] * 0.8)
        train, test = X.values[0:train_size], X.values[train_size:]
        history = [x for x in train]

        # make predictions
        predicted_mean, predicted_std = list(), list()

        for t in range(len(test)):
            history = pd.DataFrame(np.array(history))
            if t % 100 == 0:
                print('iteration', t, ' out of ', len(test))
            model = arch_model(history, mean=mean_model, p=model_param[0], q=model_param[1])
            model_fit = model.fit(disp='off', show_warning=False)
            output = model_fit.forecast(horizon=1)
            yhat_mean = output.mean.iloc[-1]
            yhat_var = output.variance.iloc[-1]
            predicted_mean.append(yhat_mean)
            predicted_std.append(np.sqrt(yhat_var))
            history = list(history.values)
            history.append(test[t])
        # calculate out of sample error
        test = np.concatenate(test, axis=0)
        error = mean_squared_error(test, predicted_mean)
        return np.sqrt(error)
