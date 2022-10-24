import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from api_key_binance import API_SECURITY, API_KEY
from models.dl.hbnn import HBNN
from models.dl.lstm import LSTM
from models.dl.lstmd import LSTMD
from models.dl.tcn import TCN
from models.dl.lstm_gru_hybrid import LSTM_GRU
from models.dl.gru import GRU
from models.stats.arima import ARIMA
from models.stats.garch import GARCH
from models.ml.rf import RF
from models.ml.knn import KNN
from models.model_probabilistic import ModelProbabilistic
from models.dl.model_probabilistic_dl import ModelProbabilisticDL
from models.dl.model_interface_dl import ModelInterfaceDL
from models.ml.svr import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, explained_variance_score, mean_squared_log_error
from tensorflow import keras
from util import dataset, plot_training, save_results, dataset_binance
from darts.metrics import mape, mse, mae
from models.dl.tft import TFT
from models.ensembles.voting_regressor import VotingRegressor
from itertools import permutations, combinations
def Average(lst):
    return sum(lst) / len(lst)

def r_squared(true, predicted):
    y = np.array(true)
    y_hat = np.array(predicted)
    
    y_m = y.mean()

    ss_res = np.sum((y-y_hat)**2)
    ss_tot = np.sum((y-y_m)**2)
    
    return 1 - (ss_res/ss_tot)

def get_all_possible_ensembles():
    iter_array= ['HYBRID', 'GRU', 'LSTM', 'TCN', 'TFT', 'RF', 'SVR', 'KNN', 'ARIMA']
    models = list()
    for i in range(1, (len(iter_array)+1)): 
        n_list = list(n for n in combinations(iter_array, i))
        for l in n_list: models.append(list(l))
    return models


def ensemble_hyperparam_test(models, clusters):
    for c in clusters:
        f_name = 'close-' + c + '-w30-h0_N'
        ensemble_name = '_'.join(models)
        print(f_name)
        en = VotingRegressor(name = ensemble_name, filename = f_name)
        en.models = models 
        en.hyperparametrization()
def train_ensemble_test(models, clusters, resources):
    for res in resources:               
        for c in clusters:
            mses, maes, rmses, mapes = [], [], [], []
            r2_scores, evs_scores, medaes, rmsles, msles = [], [], [], [], []
            all_predictions, all_labels = [], []
            f_name = res + '-' + c + '-w30-h0'
            ensemble_name = '_'.join(models)
            experiment_name = 'train-' + ensemble_name + '-' + res + '-' + c  
            vr = VotingRegressor(name = ensemble_name, filename = f_name)
            vr.models = models
            predictions, true_values = vr.predict(weighted = False, train = True)
            print("MSE", mean_squared_error(true_values, predictions))
            print("MAE", mean_absolute_error(true_values, predictions))
            print("MAPE", mean_absolute_percentage_error(true_values, predictions))
            print("RMSE", np.sqrt(mean_squared_error(true_values, predictions)))
            rmses.append(np.sqrt(mean_squared_error(true_values, predictions)))
            mapes.append(mean_absolute_percentage_error(true_values, predictions))
            mses.append(mean_squared_error(true_values, predictions))
            maes.append(mean_absolute_error(true_values, predictions))
            r2_scores.append(r_squared(true_values, predictions))
            medaes.append(median_absolute_error(true_values, predictions))
            evs_scores.append(explained_variance_score(true_values, predictions))
            save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores, evs = evs_scores, medAE = medaes, rmsle = rmsles, msle = msles)
            save_results.save_output_csv(predictions, true_values, res, experiment_name, bivariate= False)

def combine_ensemble_metrics(combine_models, res, clusters):
        for c in clusters:
            avg_maes, avg_mses, avg_rmses, avg_mapes, avg_r2 = [], [], [], [], []
            ensemble_models = []
            file_name =  'res/ensembles/metrics_train-all_ensembles' + '-' + res + '-' + c  +'.csv'
            for mod in combine_models:
                s = '_'.join(mod)
                path = 'res/ensembles/metrics_train-' + s + '-'+ res +'-' + c +'.csv'
                ensemble_models.append(s) 
                df = pd.read_csv(path)
                avg_maes.append(df['MAE'].iloc[0])
                avg_mses.append(df['MSE'].iloc[0])
                avg_rmses.append(df['RMSE'].iloc[0])
                avg_mapes.append(df['MAPE'].iloc[0])
                avg_r2.append(df['R2'].iloc[0])
            df_ensembles = pd.DataFrame(columns = ['ENSEMBLE_MODELS', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2']) 
            df_ensembles['ENSEMBLE_MODELS'] = ensemble_models
            df_ensembles['MSE']= avg_mses
            df_ensembles['RMSE'] = avg_rmses
            df_ensembles['MAE'] = avg_maes
            df_ensembles['MAPE'] = avg_mapes
            df_ensembles['R2'] = avg_r2
            df_ensembles.to_csv(file_name)

def get_ranking_of_model(model, clusters, res):
    avg_mses_with, avg_mses_without= [], []
    file_name = 'res/ensembles/ranking-' + model.upper() + '-' + res + '.csv'
    for c in clusters:
        PATH =  'res/ensembles/metrics_train-all_ensembles' + '-' + res + '-' + c  +'.csv'
        df_ensembles = pd.read_csv(PATH)
        mses_with, mses_without = [], []
        for index, row in df_ensembles.iterrows():
            arr = row['ENSEMBLE_MODELS']
            models = arr.split('_')
            if model in models: mses_with.append(row['MSE'])
            else: mses_without.append(row['MSE'])
        avg_mses_with.append(Average(mses_with))
        avg_mses_without.append(Average(mses_without))
    df_ranking = pd.DataFrame(columns = ['CRYPTO', 'AVG_MSE_WITH', 'AVG_MSE_WITHOUT'])
    df_ranking['CRYPTO'] = clusters
    df_ranking['AVG_MSE_WITH'] = avg_mses_with
    df_ranking['AVG_MSE_WITHOUT'] = avg_mses_without
    df_ranking.to_csv(file_name)


models = get_all_possible_ensembles()
for m in models: 
  print(m)
  train_ensemble_test(models = m, clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'],  resources =['close'])

train_ensemble_test(models = ['HYBRID', 'GRU', 'LSTM', 'TCN', 'TFT'], cluster = ['btc'], resources =['close'])
combine_ensemble_metrics(combine_models = models, res ='close', clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'])
all_models = ['HYBRID', 'LSTM', 'GRU', 'SVR', 'TCN', 'TFT', 'KNN', 'RF', 'ARIMA']
for m in all_models:
    get_ranking_of_model(model = m, clusters = ['btc', 'ltc', 'eth', 'xmr', 'xrp'], res ='close')