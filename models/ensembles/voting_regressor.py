import numpy as np
import pandas as pd
import optuna
from itertools import product
from sklearn.metrics import mean_squared_error
class VotingRegressor: 
    def __init__(self, name, filename):
        self.models = ['LSTM', 'GRU']
        self.parameter_list = {'weights': [(1, 0)]}
        self.p = {'weights': (1, 0)}
        self.name = name
        self.filename = filename

    def __set_possible_weights(self):
        weights = list(product(list(range(0, len(self.models))), repeat = len(self.models)))
        weights = weights[1:]
        self.parameter_list['weights'] = weights
    
    def __get_model_predictions(self):
        ml_models = ['SVR', 'RF', 'KNN']
        alt_models = ['ARIMA', 'TFT']
        el_df = pd.DataFrame(columns= self.models)
        PATH = 'res/output_'
        if train:
            PATH = 'res/output_train-' 
        for m in models: 
            m_path = PATH +  m.upper() + '-' +self.filename + '.csv'
            df = pd.read_csv(m_path)
            labels = df['labels'].values
            target_col = 'avg' + target
            target_v = df[target_col].values
            if m.upper() in ml_models:
                target_v = target_v[:len(target_v)-2]
                labels = labels[:len(labels)-2]
            elif m.upper() in alt_models: 
                target_v = target_v[2:len(target_v)-2]
                labels = labels[2:len(labels)-2]
            el_df[m] = target_v
        dct_ensemble = el_df.to_dict()
        predictions = list()
        for m in self.models: predictions.append(dct_ensemble[m.upper()])
        return predictions, labels

    def predict(self, weighted = False):
        predictions, labels = self.__get_model_predictions()
        if weighted:
            return np.average(predictions, axis = 0, weights= self.p['weights']), labels
        return np.average(predictions, axis = 0), labels
    
    def __evaluate_weights(self, trial):
        """
        Custom Function of Optuna which represents a single execution of a trial
        :param dict trial: Represents the hyperparameters being examined
        :return float mean_squared_error: MeanSquaredError of the Ensemble's Predictive Performance 
        """
        weights = trial.suggest_categorical('weights', self.parameter_list['weights'])
        predictions, lables = self.__get_model_predictions()
        weighted_preds = np.average(predictions, axis = 0, weights= weights)
        return mean_squared_error(labels, weighted_preds)
        
    
    def hyperparametrization(self):
        """
        Search the best parameter configuration using Optuna
        :return: None
        """
        self.__set_possible_weights()
        study = optuna.create_study(study_name="Voting_Regressor_Optimization", direction="minimize", sampler= optuna.samplers.GridSampler(self.parameter_list))
        study.optimize(self.__evaluate_weights, n_trials=len(self.parameter_list['weights']), show_progress_bar=True)
        print('\nBEST PARAMS: \n{}'.format(study.best_params))
        print('\nBEST VALUE:\n{}'.format(study.best_value))
        df = study.trials_dataframe()
        filename = 'talos/' + self.name +'.csv'
        df.to_csv(filename)
