from arch.__future__ import reindexing
import numpy as np
import itertools
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from arch import arch_model
from models.model_interface import ModelInterface


class GARCHPredictor():
    def __init__(self):
        ModelInterface.__init__(self, "GARCHPredictor")
        self.train_model = None
        self.model = None
        self.parameter_list = {'p': 1,
                               'q': 1,
                               'loop': 0,
                               'horizon': 1,
                               'mean': ['Constant', 'LS', 'AR'],
                               'method': ['M-H', 'BBVI'],
                               'sliding_window': 288
                               }
        self.history = None

    def training(self, X_train, y_train, X_test, y_test, p):
        X_train = pd.DataFrame(X_train)

        self.history = X_train
        if p is not None:
            self.parameter_list = p
        if self.parameter_list['sliding_window']:
            self.history = self.history.iloc[-self.parameter_list['sliding_window']:]
        if self.parameter_list['selection']:
            self.param_selection(X_train)
        self.model = arch_model(self.history, mean=self.parameter_list['mean'], p=self.parameter_list['p'],
                                q=self.parameter_list['q'])

        self.train_model = self.model.fit()

        print(self.train_model.summary())

        predictions, predicted_stdv, history = self.predict(X_test, self.parameter_list['loop'],
                                                            self.parameter_list['horizon'])

        return predictions, predicted_stdv, self.parameter_list, self.train_model

    def predict(self, X, steps, horizon):
        if self.train_model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        predicted_mean, predicted_stdv = list(), list()

        if steps == 0:
            steps = X.shape[0]
        for j in range(X.shape[0] // steps):
            t = j * steps
            self.model = arch_model(self.history, mean=self.parameter_list['mean'], p=self.parameter_list['p'],
                                    q=self.parameter_list['q'])

            self.train_model = self.model.fit()  # retrain the model at each step prediction
            output = self.train_model.forecast(horizon=steps + horizon, reindex=False)

            yhat_mean = output.mean.iloc[-1]
            yhat_var = output.variance.iloc[-1]

            [predicted_mean.append(em) for em in output.mean.iloc[-1]]
            [predicted_stdv.append(em) for em in np.sqrt(output.variance.iloc[-1])]
            obs = X[j * steps + horizon:(j + 1) * steps + horizon]
            self.history = list(self.history.values)
            [self.history.append(a) for a in X[j * steps:(j + 1) * steps]]
            
            self.history = pd.DataFrame(self.history)
            if self.parameter_list['sliding_window']:
                self.history = self.history.iloc[-self.parameter_list['sliding_window']:]
            print(t, " predicted mean = ", yhat_mean, " predicted variance = ", yhat_var, ' expected ', obs)

        # evaluate forecasts
        X = np.concatenate(X, axis=0)
        rmse = sqrt(mean_squared_error(X[:len(predicted_mean)], predicted_mean))

        self.history = list(self.history.values)
        return predicted_mean, predicted_stdv, self.history  # .train_model.params, self.history

    # evaluate an GARCH model for a given order (p,q)
    def evaluate_garch_model(self, X, model_param, mean_model):
        # prepare training dataset
        train_size = int(X.shape[0] * 0.8)
        train, test = X.values[0:train_size], X.values[train_size:]
        history = [x for x in train]
        
        # make predictions
        predicted_mean, predicted_stdv = list(), list()

        for t in range(len(test)):
            history = pd.DataFrame(np.array(history))
            if t % 100 == 0:
                print('iteration', t, ' out of ', len(test))
            model = arch_model(history, mean=mean_model, p=model_param[0], q=model_param[1])
            model_fit = model.fit()
            output = model_fit.forecast(horizon=1)
            yhat_mean = output.mean.iloc[-1]
            yhat_var = output.variance.iloc[-1]
            predicted_mean.append(yhat_mean)
            predicted_stdv.append(np.sqrt(yhat_var))
            history = list(history.values)
            history.append(test[t])
        # calculate out of sample error
        test = np.concatenate(test, axis=0)
        error = mean_squared_error(test, predicted_mean)
        return np.sqrt(error)

    def param_selection(self, X_train):
        mean_mod = ['LS', 'Constant', 'AR']
        p = q = range(1, 3)
        pq = list(itertools.product(p, q))
        ans = []
        for mean in mean_mod:
            for comb in pq:
                mod = arch_model(X_train,
                                 mean=mean,
                                 p=comb[0],
                                 q=comb[1])
                output = mod.fit()
                rmse = self.evaluate_garch_model(X_train, comb, mean)
                ans.append([comb, mean, output.aic, rmse])
                print('GARCH {} : Mean {} : AIC Calculated = {}, RMSE Calculated = {}'.format(comb, mean, output.aic,
                                                                                              rmse))
        ans_df = pd.DataFrame(ans, columns=['pq', 'mean', 'aic', 'rmse'])
        best = ans_df.loc[ans_df['rmse'].idxmin()]
        self.parameter_list['p'], self.parameter_list['q'] = best['pq']
        self.parameter_list['mean'] = best['mean']

    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.train_model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.pkl')
        self.count_save += 1

    def load_model(self, name):
        return
