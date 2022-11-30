import numpy as np
import pandas as pd
import optuna
from itertools import product
from sklearn.metrics import mean_squared_error
class VotingRegressor: 
    def __init__(self, name, filename):
        """
        Initialisation of the Voting Regressor Class
        """
        self.models = ['LSTM', 'GRU']
        """list models: List of models to combine"""
        self.parameter_list = {'weights': [(1, 0)]}
        """dict parameter_list: Dictionary of hyperparameter configuration of the model"""
        self.p = {'weights': [1, 0]}
        """dict p: Dictionary of hyperparameters search space"""
        self.name = name
        self.target = 'close'
        self.filename = filename
        self.path = 'res/output_'
        self.search_space = list()
        self.iterations = False

    def __set_possible_weights(self):
        """
        Sets the possible weight paramter based on the amount of models added        
        """
        weights = list(product(list(range(0, (len(self.models)+1))), repeat = len(self.models)))
        
        weights = weights[1:]
        self.parameter_list['weights'] = weights
        self.search_space = list(range(0, (len(weights))))
    
    def __get_model_predictions(self, train = False):
        """
        Aggregates a List of Model's Predictions
        :param train: Boolean (Default = False): determines whether to use train predictions or test predictions
        :return predictions: a 2D list of multiple models preidctions
                labels: a list of true values for evalutation
        """
        ml_models = ['SVR', 'RF', 'KNN', 'ARIMA', 'TFT']
        el_df = pd.DataFrame(columns= self.models)
       
        dct_ensemble = dict.fromkeys(self.models, None)
        length_difference = 0
        start_at = 0
        tft_present = False
        if train:
            self.path = 'res/train/output_train-' 
            start_at = 1
       
        for m in self.models: 
            m_path = self.path +  m.upper() + '-' +self.filename + '.csv'
            df = pd.read_csv(m_path)
            labels = df['labels'].values
            target_col = self.target
            target_v = df[target_col].values
            if m.upper() in ml_models:
                l = length_difference
                target_v = target_v[:len(target_v)-l]
                labels = labels[:len(labels)-l]
            
            target_v = target_v[start_at:]
            labels = labels[start_at:]
            dct_ensemble[m.upper()] = target_v
        
        predictions = list()
        for m in self.models: 
            predictions.append(dct_ensemble[m.upper()])
        
        return predictions, labels
  
    def __get_iterative_model_predictions(self, index):
        """
        Aggregates a List of Model's Predictions
        :param train: Boolean (Default = False): determines whether to use train predictions or test predictions
        :return predictions: a 2D list of multiple models preidctions
                labels: a list of true values for evalutation
        """
        ml_models = ['SVR', 'KNN', 'ARIMA']
        change_length = [ 'ARIMA']
        el_df = pd.DataFrame(columns= self.models)
        dct_ensemble = dict.fromkeys(self.models, None)
       
        for m in self.models: 
            m_file =  m.upper() + '-' +self.filename + '.csv'
            l_path =  self.path + 'labels_' + m_file
            df = pd.read_csv(l_path)
            loc = index
            if m in ml_models:
                loc = 0 
            i_path = self.path + 'preds_' + m_file
            
            labels  =df.iloc[loc].values
            labels = labels[1:]
            df_iteration = pd.read_csv(i_path)
            pred =df_iteration.iloc[loc].values
            preds = pred[1:]
            if m in change_length:
                preds = preds[:-5]
                labels = labels[:-5]
            dct_ensemble[m.upper()] = preds
        
        predictions = list()
        for m in self.models: 
            predictions.append(dct_ensemble[m.upper()])
        
        return predictions, labels
    def get_predictions(self, index):
        preds, labels = self.__get_iterative_model_predictions(index = index)
        return preds, labels
    def predict(self, train = False, weighted = False, index = -1):
        """
        Inference step on the samples
        :param:  boolean train (Default = False): Determines whether to use train or test model outputs
                 boolean weighted (Default = False): Details whether to use weighted average
        :return: np.array combined_predictions: array of combined model predictions
                 np.array labels: array of true values 
        """
        if index == -1:
            predictions, labels = self.__get_model_predictions(train)
        else: 
            predictions, labels = self.__get_iterative_model_predictions(index = index)
        labels = np.asarray(labels)
        if weighted:
            
            combined_predictions = np.average(predictions, axis = 0, weights= self.p['weights'])
        else:
            combined_predictions = np.average(predictions, axis = 0)
        return  combined_predictions, labels
    
    def __evaluate_weights(self, weight):
        """
        Custom Function of Optuna which represents a single execution of a trial
        :param dict trial: Represents the hyperparameters being examined
        :return float mean_squared_error: MeanSquaredError of the Ensemble's Predictive Performance 
        """
        predictions, labels = self.__get_model_predictions(train = True)
        weighted_preds = np.average(predictions, axis = 0, weights= weight)
        return predictions, weighted_preds, mean_squared_error(labels, weighted_preds)
        
    
    def hyperparametrization(self):
        """
        Search the best parameter configuration using Optuna
        :return: None
        """
        self.__set_possible_weights()
        df = pd.DataFrame(columns = ['Predictions', 'Weights', 'Weighted_Predictions', 'MSE'])
        mses = []
        weights  = []
        predictions =[]
        w_predictions =[]
        print('Hyperparameterisation in Progress')
        """
        Infinite Looping Occurs with 
        for i in self.search_space:
            w = self.parameter_list['weights'][i]
            preds, w_preds, error = self.__evaluate_weights(w)
            mses.append(error)
            predictions.append(preds)
            w_predictions.append(w_preds)
            weights.append(w)
        """
        print('Hyperparameterisation Completed')
        df['Weights'] = weights
        df['MSE'] =mses
        df['Predictions'] = predictions
        df['Weighted_Predictions'] = w_predictions
        df.to_csv('res/Ensemble_hyperparam_output.csv')

        
