
from cProfile import label
import glob
from importlib.machinery import OPTIMIZED_BYTECODE_SUFFIXES
from pyexpat import model
from statistics import mode
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras
from util import dataset, plot_training, save_results, dataset_binance
from darts.metrics import mape, mse, mae
from models.dl.tft import TFT



ITERATIONS = 10
apiKey = API_KEY
apiSecurity = API_SECURITY
dl_names = ['HYBRID', 'LSTM', 'GRU', 'TCN']
ml_names = ['RF', 'SVR', 'KNN', 'ARIMA']
stats_names = ['ARIMA']
def Average(lst):
    return sum(lst) / len(lst)

def dataset_test(wins, horizons, resources, clusters):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes = [], []
                    #experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                    # Data creation and load
                   
                    ds = dataset.DatasetInterface(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.8)

                    ds.dataset_creation()
                    ds.dataset_normalization(['minmax'])  # , 'standard'])
                    ds.data_summary()
 

def dataset_binance_test(wins, horizons, resources, clusters):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes = [], []
                    #experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                    # Data creation and load
                   
                    ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.8, apiKey= apiKey, apiSecurity= apiSecurity)
                    
                    ds.build_crypto_dataset(f"{c.upper()}USD", '2015', '2020', f"{c.upper()}USDT", '1 Jan, 2013', '31 May, 2022')
                    ds.dataset_creation()
                    ds.dataset_normalization(['minmax'])  # , 'standard'])
                    ds.data_summary()


def dataset_binance_test_2(wins, horizons, resources, clusters):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes = [], []
                    #experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                    # Data creation and load
                   
                    ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.8, apiKey= apiKey, apiSecurity= apiSecurity)
                    df = ds.get_binance_data('LTCUSDT', '1 Jan, 2018', '23 April, 2022')
                    ds.save_to_csv(df)
                    ds.dataset_creation()
                    ds.dataset_normalization(['minmax'])  # , 'standard'])
                    ds.data_summary()

def total_test(wins, horizons, resources, clusters, model_name, scaling, output = 0):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes, rmses, mapes = [], [], [], []
                    predictions, true_values =  [], []
                    for i in range(ITERATIONS):
                        experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                        # Data creation and load
                        # ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                        #                               horizon=h, training_features=['avgcpu', 'time', 'avgmem'],
                        #                               target_name=['avg' + res, 'avgmem'], train_split_factor=0.8)

                        ds = dataset.DatasetInterface(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                    horizon=h, training_features=['close'],
                                                    target_name=['close'], train_split_factor=0.9)

                        ds.dataset_creation()
                        ds.dataset_normalization(scaling)  # , 'minmax', 'minmax'])


                        ds.data_summary()
                        parameters = pd.read_csv("hyperparams/p_hbnn-" + 'a' + ".csv").iloc[0]

                        files = sorted(
                            glob.glob("saved_models/talos-HBNN-" + c + "-cpu-w" + str(win) + "-h" + str(h) + "*_weights.tf.i*"))

                        dense_act = 'relu'
                        if 'relu' in parameters['first_dense_activation']:
                            dense_act = 'relu'
                        elif 'tanh' in parameters['first_dense_activation']:
                            dense_act = 'tanh'

                        # HBNN
                        
                        if model_name == 'LSTM':
                            p =  {'first_conv_dim': 64,
                                'first_conv_activation': 'relu',
                                'first_conv_kernel': 5,
                                'first_lstm_dim': 75,
                                'first_dense_dim': 16,
                                'first_dense_activation':'relu',
                                'batch_size': 256,
                                'patience': 50,
                                'epochs': 200,  
                                'optimizer': 'adam',
                                'lr':1E-4,
                                'momentum': 0.9,
                                'decay': 1E-3,
                                }
                        elif model_name == 'GRU':
                            p = {'first_gru_dim': 75,
                                    'gru_activation': 'relu',
                                    'first_dense_dim': 100,
                                    'first_dense_activation': 'relu',
                                    'dense_kernel_init': 'he_normal',
                                    'batch_size': 256,
                                    'epochs': 200,
                                    'patience': 50,
                                    'optimizer': 'adam',
                                    'lr': 1E-3,
                                    'momentum': 0.9,
                                    'decay': 1E-3,
                        }
                        elif model_name =='TCN':
                            p = {'conv_filter': 32,
                                'conv_kernel': 16,
                                'conv_activation': 'relu',
                                'dropout_rate': 0.05,
                                'dense_dim': 64,
                                'dilation_rate': 8,
                                'dense_activation': 'relu',
                                'dense_kernel_init': 'he_normal',
                                'batch_size': 256,
                                'epochs': 200,
                                'patience': 50,
                                'optimizer': 'adam',
                                'lr': 1E-4,
                                'momentum': 0.9,
                                'decay': 1E-4,
                    }
                        elif model_name =='HYBRID':
                            p = {
                                'lstm_dim_1': 75,
                                'lstm_activation': 'relu',
                                'dropout_rate_1': 0.05, 
                                'lstm_dim_2':  50,
                                'dense_dim_1':  32,
                                'dense_activation': 'relu',
                                'dense_kernel_init': 'he_normal',
                                'gru_dim':50,
                                'gru_activation':'relu',
                                'dropout_rate_2':  0.0,
                                'dense_dim_2':  64,
                                'batch_size': 256,
                                'epochs': 200,
                                'patience': 50,
                                'optimizer': 'adam',
                                'lr': 1E-3,
                                'momentum': 0.9,
                                'decay': 1E-4,
                    }
                        # # ARIMA
                        elif model_name == 'ARIMA':
                            p = {'p': 1,
                                'd': 0,
                                'q': 2,
                                'P': 2,
                                'Q': 0,
                                'D': 0,
                                'S': 12,
                                'loop': 0,
                                'horizon': 0,
                                }
                
                        elif model_name == 'GARCH':
                            # # GARCH
                            p = {'p': 2,
                                'q': 1,
                                'loop': 0,
                                'horizon': 2,
                                'mean': 'LS',
                                }
                        elif model_name == 'SVR':
                            # # SVR
                            p = {'kernel': 'poly',
                                'degree': 2,
                                'gamma': 'auto',
                                'tol': 0.001,
                                'C': 10,
                                }
                        elif model_name =="KNN":
                            p = {'n_neighbors': 2,
                                'weights': 'uniform',
                                'algorithm': 'auto',
                                'p': 1,
                                
                                }
                        elif model_name == 'RF':
                            p = {'n_estimators': 500,
                                'criterion': "mae",
                                'max_depth': 10,
                                'max_features': "log2",
                                'bootstrap': True,
                                }

                    
                        print("RESOURCE:", res, "CLUSTER:", c, "HORIZON:", h, "WIN:", win)
                        if model_name == 'HBNN':
                            model = HBNN(experiment_name)
                        elif model_name == 'LSTM':
                            model = LSTM(experiment_name)
                        elif model_name == 'LSTMD':
                            model = LSTMD(experiment_name)
                        elif model_name == 'ARIMA':
                            model = ARIMA(experiment_name)   
                        elif model_name == 'GARCH':
                            model = GARCH(experiment_name)
                        elif model_name == 'SVR':
                            model = SVR(experiment_name)
                        elif model_name == 'RF':
                            model = RF(experiment_name)
                        elif model_name == 'KNN':
                            model = KNN(experiment_name)
                        elif model_name == 'HYBRID':
                            model = LSTM_GRU(experiment_name)
                        elif model_name == 'GRU':
                            model = GRU(experiment_name)
                        elif model_name == 'TCN':
                            model = TCN(experiment_name)    
                        
                        
                        if model_name == 'LSTM' or model_name == 'SVR' or model_name == 'KNN' or model_name == 'RF' or model_name == 'GRU' or model_name == 'HYBRID' or model_name == 'TCN':
                            model.ds = ds
                            model.p = p
                            model.create_model()
                        
                            model.fit()
                            print("Training complete")
                            
                            if output> 0:
                                if model_name in ml_names: 
                                    n = len(ds.X_test_array) - output
                                    y= len(ds.y_test_array) - output
                                    to_predict = ds.X_test_array[n:]
                                else:
                                    n = len(ds.X_test) - output
                                    y= len(ds.y_test) - output
                                    to_predict = ds.X_test[n:]
                            else:
                                to_predict = ds.X_test
                                n = 0
                                y = 0
                            train_mean = model.evaluate()
                            preds = model.predict(to_predict)

                            if h > 0:
                                preds = preds[:-h]

                            if len(ds.target_name) <= 1:
                                labels = ds.y_test_array[(h+y):(len(preds)+h+y)].reshape(-1, 1)
                                train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
                            else:
                                labels = ds.y_test_array[h:len(preds)+h]
                                train_labels = ds.y_train_array[h:len(train_mean) + h]

                            print("MSE", mean_squared_error(labels, preds))
                            print("MAE", mean_absolute_error(labels, preds))
                            print("MAPE", mean_absolute_percentage_error(labels, preds))
                            print("RMSE", np.sqrt(mean_squared_error(labels, preds)))
                            rmses.append(np.sqrt(mean_squared_error(labels, preds)))
                            mapes.append(mean_absolute_percentage_error(labels, preds))
                            mses.append(mean_squared_error(labels, preds))
                            maes.append(mean_absolute_error(labels, preds))
                            predictions.append(preds)
                            true_values.append(labels)
                       

                            if len(ds.target_name) <= 1:
                                train_mean = np.array(train_mean).reshape(-1, 1)
                                train_mean = np.concatenate(train_mean, axis=0)
                                # train_labels = np.concatenate(ds.y_train.reshape(-1, 1), axis=0)
                                if isinstance(model, ModelInterfaceDL):
                                    preds = np.array(preds).reshape(-1, 1)
                                    preds = np.concatenate(preds, axis=0)
                                    labels = np.concatenate(labels, axis=0)

                            save_results.save_output_csv(preds, labels, 'avg' + res, model.name,
                                                        bivariate=len(ds.target_name) > 1)
                            save_results.save_output_csv(train_mean, train_labels, 'avg' + res, 'train-' + model.name,
                                                        bivariate=len(ds.target_name) > 1)

                            save_results.save_params_csv(model.p, model.name)

                        elif model_name == 'ARIMA' or model_name =='GARCH':
                                model.ds = ds
                                labels = ds.ts_test.pd_dataframe()
                                labels = np.array(labels.values).reshape(-1, 1)
                                labels = np.concatenate(labels, axis=0)
                                if output > 0:
                                    n = len(ds.ts_test) - output
                                    labels = labels[n:]
                                to_predict = labels
                                preds = list()
                            
                                model.fit()
                                if model_name =='ARIMA':
                                    yhat= model.predict(to_predict)
                                else:
                                    y_hat= model.temp_model.forecas(to_predict)
                                preds= yhat[0]
                                
                                predictions.append(preds)
                                true_values.append(labels)        
                                save_results.save_output_csv(preds, labels, 'avg' + res, model.name,
                                                            bivariate=len(ds.target_name) > 1)
                                rmses.append(np.sqrt(mean_squared_error(labels, preds)))
                                mapes.append(mean_absolute_percentage_error(labels, preds))
                                mses.append(mean_squared_error(labels, preds))
                                maes.append(mean_absolute_error(labels, preds))

                    optimal_index = mapes.index(min(mapes))
                    avg_mae = Average(maes)
                    avg_mse = Average(mses)
                    avg_rmse = Average(rmses)
                    avg_mape = Average(mapes)
                    rmses.append(avg_rmse)
                    mapes.append(avg_mape)
                    mses.append(avg_mse)
                    maes.append(avg_mae)
                    print("__________AVERAGE______________")
                    print("MSE", avg_mse)
                    print("MAE", avg_mae)
                    print("MAPE", avg_mape)
                    print("RMSE", avg_rmse)
                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmse = rmses, mape = mapes, filename = experiment_name)
                    save_results.save_output_csv(predictions[optimal_index], true_values[optimal_index], 'avg' + res, model.name,
                                                    bivariate=len(ds.target_name) > 1)
                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmse = rmses, mape = mapes, filename = experiment_name)
                       
                  
                    
                        


def hyper_param_test(wins, horizons, resources, clusters, model_name, scaling):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes = [], []
                    experiment_name = model_name + '-' + res + '-'+'param-' + c + '-w' + str(win) + '-h' + str(h)

                    ds = dataset.DatasetInterface(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.8)

                    ds.dataset_creation()
                    ds.dataset_normalization(scaling)  # , 'minmax', 'minmax'])

                  
                    ds.data_summary()
                   

    
                    if model_name == 'ARIMA':
                        model = ARIMA(experiment_name)
                    elif model_name =='GARCH':
                        model = GARCH(experiment_name)
                    elif model_name == 'RF':
                        model =  RF(experiment_name)
                        model.parameter_list = {'n_estimators': [150, 180, 200, 250, 300, 350, 400, 500],
                               'criterion': ["mse", "mae", "poisson"],
                               'max_depth': [None, 10, 20, 50, 70, 100],
                               'max_features': ["auto", "sqrt" ,"log2"],
                               'bootstrap': [True, False]
                               }
                    elif model_name == 'SVR':
                         model = SVR(experiment_name)
                         model.parameter_list = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                               'degree': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                               'gamma' : ['auto', 0.1, 1, 0.0, 0.001],
                               'C': [0.1,1, 10, 100],

                               }
                    elif model_name =='KNN':
                        model = KNN(experiment_name)
                        model.parameter_list = {'n_neighbors': list(range(1,30)),
                               'weights': ('uniform', 'distance'),
                               'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                               'p': [1, 2]
                               }
                        
                    elif model_name =='LSTM':
                        model =LSTM(experiment_name)
                        model.parameter_list ={
                                'first_conv_dim': [32, 64],
                               'first_conv_kernel': [3, 5, 7],
                               'first_conv_activation': ['relu'],
                               'first_lstm_dim': [50, 75],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': ['relu', 'tanh'],
                               'batch_size': [256],
                               'epochs': [200],
                               'patience': [20],
                               'optimizer': ['adam'],
                               'lr': [1E-3, 1E-4, 1E-5],
                              
                        }
                    elif model_name == 'TCN':
                        model = TCN(experiment_name)
                        model.parameter_list = {
                                                'conv_filter': [16, 32, 64],
                                                'conv_kernel': [4, 8, 16],
                                                'conv_activation': ['relu'],
                                                'dilation_rate': [1, 4, 8],
                                                'dropout_rate': [0.0 , 0.05 , 0.1],
                                                'dense_dim': [32, 64],
                                                'dense_activation': ['relu'],
                                                'batch_size': [256],
                                                'epochs': [200],
                                                'patience': [20],
                                                'optimizer': ['adam'],
                                                'lr': [1E-3, 1E-4, 1E-5],
                                               
                               }
                        
                    elif model_name =='HYBRID':
                        model =LSTM_GRU(experiment_name)
                        model.parameter_list ={
                                            
                                                'lstm_dim_1': [50, 75],
                                                'lstm_activation': ['relu'],
                                                'dropout_rate_1': [0.05, 0.1],
                                                'lstm_dim_2': [50, 75],
                                                'dense_dim_1': [32, 64],
                                                'dense_activation': ['relu'],
                                                'gru_dim':[50,75],
                                                'gru_activation': ['relu'],
                                                'dropout_rate_2': [0.0, 0.05, 0.1],
                                                'dense_dim_2': [ 32, 64],
                                                'epochs': [200],
                                                'patience': [20],
                                                'batch_size': [256],
                                                'optimizer': ['adam'],
                                                'lr': [1E-3, 1E-4, 1E-5],
                                          
                               
                        }
                    elif model_name =='GRU':
                        model = GRU(experiment_name)
                        model.parameter_list = {
                                'first_gru_dim': [50, 75, 100],
                               'gru_activation': ['relu'],
                               'first_dense_dim': [16, 32, 64, 100],
                               'first_dense_activation': ['relu'],
                               'batch_size': [256],
                               'epochs': [200],
                               'patience': [20],
                               'optimizer': ['adam', 'nadam','rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               }
                    model.ds = ds
                    model.verbose = 2
                    model.create_model()
                    model.hyperparametrization()




def tft_test(wins, horizons, resources, clusters, model_name, scaling, output = 0):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes, rmses, mapes = [], [], [], []
                    predictions, true_values = [], []
                    for i in range(ITERATIONS):
                        
                        experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)
                        ds = dataset.DatasetInterface(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                    horizon=h, training_features=['close'],
                                                    target_name=['close'], train_split_factor=0.9)

                        ds.dataset_creation()
                        ds.dataset_normalization(scaling)
                        ds.data_summary()
                    
                        labels = ds.ts_test.pd_dataframe()
                        labels = np.array(labels.values).reshape(-1, 1)
                        labels = np.concatenate(labels, axis=0)
                        if output > 0:
                            n = len(ds.ts_test) - output
                            labels = labels[n:]
                        
                        if model_name =='TFT':
                            model = TFT(experiment_name)
                            model.p = {
                                'epochs': 200, 
                                'input_chunk_length': 30,
                                'hidden_layer_dim': 64,
                                'num_lstm_layers' : 3,
                                'num_attention_heads': 7,
                                'dropout_rate': 0.05,
                                'batch_size': 256,
                                'output_chunk_length': 1,
                                'patience': 50,
                                'lr': 1e-3,
                                'optimizer': 'adam',
                                'feed_forward': 'GatedResidualNetwork',
                        
                        }
                        print('PARAMS: ', model.p)
                        model.create_model()
                        model.ds = ds
                        model.fit()
                        train_mean = model.evaluate()
                        
                        
                        
                            
                        
                        to_predict = labels
                        preds = model.predict(to_predict)
                        preds = preds.pd_dataframe()
                        preds = np.array(preds.values).reshape(-1, 1)
                        preds = np.concatenate(preds, axis=0)
                        predictions.append(preds)
                        true_values.append(labels)
                        

                        print("MSE", mean_squared_error(labels, preds))
                        print("MAE", mean_absolute_error(labels, preds))
                        print("MAPE", mean_absolute_percentage_error(labels, preds))
                        print("RMSE", np.sqrt(mean_squared_error(labels, preds)))
                        rmses.append(np.sqrt(mean_squared_error(labels, preds)))
                        mapes.append(mean_absolute_percentage_error(labels, preds))
                        mses.append(mean_squared_error(labels, preds))
                        maes.append(mean_absolute_error(labels, preds))


                    optimal_index = mapes.index(min(mapes))
                    avg_mae = Average(maes)
                    avg_mse = Average(mses)
                    avg_rmse = Average(rmses)
                    avg_mape = Average(mapes)
                    rmses.append(avg_rmse)
                    mapes.append(avg_mape)
                    mses.append(avg_mse)
                    maes.append(avg_mae)
                    print("__________AVERAGE______________")
                    print("MSE", avg_mse)
                    print("MAE", avg_mae)
                    print("MAPE", avg_mape)
                    print("RMSE", avg_rmse)
                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmse = rmses, mape = mapes, filename = experiment_name)
                    save_results.save_output_csv(predictions[optimal_index], true_values[optimal_index], 'avg' + res, model.name,
                                                    bivariate=len(ds.target_name) > 1)
                    #save_results.save_output_csv(train_mean, train_labels, 'avg' + res, 'train-' + model.name,
                                                    #bivariate=len(ds.target_name) > 1)

                    save_results.save_params_csv(model.p, model.name)
                
def tft_combined_test(wins, horizons, resources, clusters, model_name, scaling, pred_crypto):
  
    for win in wins:
        for res in resources:
            for h in horizons:
                mses, maes, rmses, mapes = [], [], [], []
                experiment_name = model_name + '-' + res + '-' + pred_crypto + '-w' + str(win) + '-h' + str(h)
                if model_name =='TFT':
                        model = TFT(experiment_name)
                        model.p = {
                            'epochs': 200, 
                            'input_chunk_length': 30,
                            'hidden_layer_dim': 128,
                            'num_lstm_layers' : 5,
                            'num_attention_heads': 5,
                            'dropout_rate': 0.05,
                            'batch_size': 256,
                            'output_chunk_length': 1,
                            'patience': 50,
                            'lr': 1e-3,
                            'optimizer': 'adam',
                    
                    }
                
                model.create_model()
                ds = dataset.DatasetInterface(filename='crypto_task_' + clusters[0] + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.8)

                ds.dataset_creation()
                ds.dataset_normalization(scaling)
                ds.data_summary()
                model.ds = ds
                model.fit()
                for i in range(1, len(clusters)):
                    c = clusters[i]
                    ds = dataset.DatasetInterface(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.8)

                    ds.dataset_creation()
                    ds.dataset_normalization(scaling)
                    ds.data_summary()
                    model.load_model()
                    model.ds = ds
                    model.fit()
                ds = dataset.DatasetInterface(filename='crypto_task_' + pred_crypto + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.8)

                ds.dataset_creation()
                ds.dataset_normalization(scaling)
                ds.data_summary()  
                model.ds = ds
                print(model.count_save)
                model.count_save = model.count_save - 1
                model.load_model()
                model.fit()
                preds = model.predict(ds.ts_test)
                print('MSE: ', mse(ds.ts_test, preds))
                print('MAE: ', mae(ds.ts_test, preds))
                print('RMSE: ', np.sqrt(mse(ds.ts_test, preds)))
                print('MAPE: ', (mape(ds.ts_test, preds)/100))
                mses.append(mse(ds.ts_test, preds))
                rmses.append(np.sqrt(mse(ds.ts_test, preds)))
                maes.append(mae(ds.ts_test, preds))
                mapes.append(( mape(ds.ts_test, preds)/100))
                save_results.save_metrics_csv(mses=mses, maes =  maes,rmse = rmses, mape = mapes, filename = experiment_name)        
    
                    
                    

def tft_hyperparam_test(wins, horizons, resources, clusters, model_name, scaling):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes, rmses, mapes = [], [], [], []
                    experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                    # Data creation and load
                    # ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                    #                               horizon=h, training_features=['avgcpu', 'time', 'avgmem'],
                    #                               target_name=['avg' + res, 'avgmem'], train_split_factor=0.8)

                    ds = dataset.DatasetInterface(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.9)

                    ds.dataset_creation()
                    ds.dataset_normalization(scaling)
                    ds.data_summary()
                    model = TFT(experiment_name)
                    model.parameter_list = {
                                        "input_chunk_length":[30], 
                                        "hidden_size":[16, 32, 64, 128], 
                                        "lstm_layers":[1, 2, 3, 4], 
                                        "num_attention_heads":[4, 5, 6, 7], 
                                        "dropout":[0.0, 0.05, 0.1], 
                                        "batch_size":[256], 
                                        'output_chunk_length': [1],
                                        "n_epochs":[1000],
                                        "lr": [1e-3, 1e-4, 1e-5],
                                        'optimizer': ['adam'],      
                                        "feed_forward": ['GatedResidualNetwork']  , 
                                            }
                    model.ds = ds
                    model.create_model()
                    model.hyperparametrization()
                    


def first_diff_dataset_test(wins, horizons, resources, clusters, model_name, scaling):
       for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes = [], []
                    #experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                    # Data creation and load
                   
                    ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                horizon=h, training_features=['close'],
                                                target_name=['close'], train_split_factor=0.9, apiKey= apiKey, apiSecurity= apiSecurity)
                    df, diff_df = ds.differenced_dataset()
                    ds.df = diff_df
                    print(ds.df)
                    ds.dataset_creation(detrended=True)
                    ds.dataset_normalization(['minmax'])  # , 'standard'])
                    ds.data_summary()
                    #invert scaling
                    #invert ts
                    labels = ds.ts_test.pd_dataframe()
                    labels = np.array(labels.values).reshape(-1, 1)
                    ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain =ds.get_ts_data(df=df)
                    preds = ds.inverse_transform_predictions(preds= labels, method=scaling[0], X= ts_ttrain)
                    invert_preds = ds.inverse_differenced_dataset(df=df, diff_vals=preds)
                    print(invert_preds)
                    #invert np arrays
                    np_labels = ds.y_test_array
                    np_labels = np.array(np_labels).reshape(-1, 1)
                    np_preds =  ds.inverse_transform_predictions(preds = np_labels)
                    invert_np_preds = ds.inverse_differenced_dataset(df=df, diff_vals = np_preds)
                    print(invert_np_preds)
def first_diff_tft_test(wins, horizons, resources, clusters, model_name, scaling, output = 0):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes, rmses, mapes = [], [], [], []
                    predictions, true_values =[], []
                    for i in range(ITERATIONS):
                            experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)
                            ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                                        horizon=h, training_features=['close'],
                                                                        target_name=['close'], train_split_factor=0.9, apiKey= apiKey, apiSecurity= apiSecurity)
                            df, diff_df = ds.differenced_dataset()
                            ds.df = diff_df
                                        
                            ds.dataset_creation(detrended=True)
                            ds.dataset_normalization(scaling)  # , 'standard'])
                            ds.data_summary()
                            if model_name =='TFT':
                                model = TFT(experiment_name)
                                model.p = {
                                    'epochs': 200, 
                                    'input_chunk_length': 30,
                                    'hidden_layer_dim': 64,
                                    'num_lstm_layers' : 3,
                                    'num_attention_heads': 7,
                                    'dropout_rate': 0.05,
                                    'batch_size': 256,
                                    'output_chunk_length': 1,
                                    'patience': 50,
                                    'lr': 1e-3,
                                    'optimizer': 'adam',
                                    'feed_forward': 'GatedResidualNetwork',
                            
                                }
                                print('PARAMS: ', model.p)
                                model.create_model()
                                model.ds = ds
                                model.fit()
                                
                                to_predict= ds.ts_test.pd_dataframe()
                                to_predict= np.array(to_predict.values).reshape(-1, 1)
                                if output > 0:
                                    n = len(ds.ts_test) - output
                                    to_predict = to_predict[n:]
                                ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain =ds.get_ts_data(df=df)            
                                            
                                            
                                                
                                            
                                
                                preds = model.predict(to_predict)
                                preds = preds.pd_dataframe()
                                preds = np.array(preds.values).reshape(-1, 1)
                                
                                preds = ds.inverse_transform_predictions(preds= preds, method=scaling[0], X= ts_ttrain)
                                preds= ds.inverse_differenced_dataset(df=df, diff_vals=preds)

                                            
                                labels = ts_test.pd_dataframe()
                                labels = np.array(labels.values).reshape(-1, 1)
                                if output > 0:
                                    n = len(ds.ts_test) - output
                                    labels = labels[n:]
                                print("MSE", mean_squared_error(labels, preds))
                                print("MAE", mean_absolute_error(labels, preds))
                                print("MAPE", mean_absolute_percentage_error(labels, preds))
                                print("RMSE", np.sqrt(mean_squared_error(labels, preds)))
                                rmses.append(np.sqrt(mean_squared_error(labels, preds)))
                                mapes.append(mean_absolute_percentage_error(labels, preds))
                                mses.append(mean_squared_error(labels, preds))
                                maes.append(mean_absolute_error(labels, preds))
                                predictions.append(preds)
                                true_values.append(labels)

                    optimal_index = mapes.index(min(mapes))
                    avg_mae = Average(maes)
                    avg_mse = Average(mses)
                    avg_rmse = Average(rmses)
                    avg_mape = Average(mapes)
                    rmses.append(avg_rmse)
                    mapes.append(avg_mape)
                    mses.append(avg_mse)
                    maes.append(avg_mae)
                    print("__________AVERAGE______________")
                    print("MSE", avg_mse)
                    print("MAE", avg_mae)
                    print("MAPE", avg_mape)
                    print("RMSE", avg_rmse)
                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmse = rmses, mape = mapes, filename = experiment_name)
                    save_results.save_output_csv(predictions[optimal_index], true_values[optimal_index], 'avg' + res, model.name,
                                                    bivariate=len(ds.target_name) > 1)
                    
def first_diff_total_test(wins, horizons, resources, clusters, model_name, scaling, output = 0):
     
     for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    mses, maes, rmses, mapes = [], [], [], []
                    predictions, true_values = [], []
                    for i in range(ITERATIONS):
                        
                        experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                        # Data creation and load
                        # ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                        #                               horizon=h, training_features=['avgcpu', 'time', 'avgmem'],
                        #                               target_name=['avg' + res, 'avgmem'], train_split_factor=0.8)

                        ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                    horizon=h, training_features=['close'],
                                                    target_name=['close'], train_split_factor=0.9, apiKey= apiKey, apiSecurity= apiSecurity)
                        df, diff_df = ds.differenced_dataset()
                        ds.df = diff_df
                        ds.dataset_creation(detrended=True)
                        
                        ds.dataset_normalization(scaling)
                    

                        ds.data_summary()
                        parameters = pd.read_csv("hyperparams/p_hbnn-" + 'a' + ".csv").iloc[0]

                        files = sorted(
                            glob.glob("saved_models/talos-HBNN-" + c + "-cpu-w" + str(win) + "-h" + str(h) + "*_weights.tf.i*"))

                        dense_act = 'relu'
                        if 'relu' in parameters['first_dense_activation']:
                            dense_act = 'relu'
                        elif 'tanh' in parameters['first_dense_activation']:
                            dense_act = 'tanh'

                        # HBNN
                        
                        if model_name == 'LSTM':
                            p =  {'first_conv_dim': 64,
                                'first_conv_activation': 'relu',
                                'first_conv_kernel': 5,
                                'first_lstm_dim': 75,
                                'first_dense_dim': 16,
                                'first_dense_activation':'relu',
                                'batch_size': 256,
                                'patience': 50,
                                'epochs': 200,  
                                'optimizer': 'adam',
                                'lr':1E-4,
                                'momentum': 0.9,
                                'decay': 1E-3,
                                }
                        elif model_name == 'GRU':
                            p = {'first_gru_dim': 75,
                                    'gru_activation': 'relu',
                                    'first_dense_dim': 100,
                                    'first_dense_activation': 'relu',
                                    'dense_kernel_init': 'he_normal',
                                    'batch_size': 256,
                                    'epochs': 200,
                                    'patience': 50,
                                    'optimizer': 'adam',
                                    'lr': 1E-3,
                                    'momentum': 0.9,
                                    'decay': 1E-3,
                        }
                        elif model_name =='TCN':
                            p = {'conv_filter': 32,
                                'conv_kernel': 16,
                                'conv_activation': 'relu',
                                'dropout_rate': 0.05,
                                'dense_dim': 64,
                                'dilation_rate': 8,
                                'dense_activation': 'relu',
                                'dense_kernel_init': 'he_normal',
                                'batch_size': 256,
                                'epochs': 200,
                                'patience': 50,
                                'optimizer': 'adam',
                                'lr': 1E-4,
                                'momentum': 0.9,
                                'decay': 1E-4,
                            }
                        elif model_name =='HYBRID':
                            p = {
                                'lstm_dim_1': 75,
                                'lstm_activation': 'relu',
                                'dropout_rate_1': 0.05, 
                                'lstm_dim_2':  50,
                                'dense_dim_1':  32,
                                'dense_activation': 'relu',
                                'dense_kernel_init': 'he_normal',
                                'gru_dim':50,
                                'gru_activation':'relu',
                                'dropout_rate_2':  0.0,
                                'dense_dim_2':  64,
                                'batch_size': 256,
                                'epochs': 200,
                                'patience': 50,
                                'optimizer': 'adam',
                                'lr': 1E-3,
                                'momentum': 0.9,
                                'decay': 1E-4,
                        }
                        # # ARIMA
                        elif model_name == 'ARIMA':
                            p = {'p': 1,
                                'd': 0,
                                'q': 2,
                                'P': 2,
                                'Q': 0,
                                'D': 0,
                                'S': 12,
                                'loop': 0,
                                'horizon': 0,
                                }
                
                        elif model_name == 'GARCH':
                            # # GARCH
                            p = {'p': 2,
                                'q': 1,
                                'loop': 0,
                                'horizon': 2,
                                'mean': 'LS',
                                }
                        elif model_name == 'SVR':
                            # # SVR
                            p = {'kernel': 'poly',
                                'degree': 2,
                                'gamma': 'auto',
                                'tol': 0.001,
                                'C': 10,
                                }
                        elif model_name =="KNN":
                            p = {'n_neighbors': 2,
                                'weights': 'uniform',
                                'algorithm': 'auto',
                                'p': 1,
                                
                                }
                        elif model_name == 'RF':
                            p = {'n_estimators': 500,
                                'criterion': "mae",
                                'max_depth': 10,
                                'max_features': "log2",
                                'bootstrap': True,
                                }

                    
                        print("RESOURCE:", res, "CLUSTER:", c, "HORIZON:", h, "WIN:", win)
                        if model_name == 'HBNN':
                            model = HBNN(experiment_name)
                        elif model_name == 'LSTM':
                            model = LSTM(experiment_name)
                        elif model_name == 'LSTMD':
                            model = LSTMD(experiment_name)
                        elif model_name == 'ARIMA':
                            model = ARIMA(experiment_name)
                            
                        elif model_name == 'GARCH':
                            model = GARCH(experiment_name)
                        elif model_name == 'SVR':
                            model = SVR(experiment_name)
                        elif model_name == 'RF':
                            model = RF(experiment_name)
                        elif model_name == 'KNN':
                            model = KNN(experiment_name)
                        elif model_name == 'HYBRID':
                            model = LSTM_GRU(experiment_name)
                        elif model_name == 'GRU':
                            model = GRU(experiment_name)
                        elif model_name == 'TCN':
                            model = TCN(experiment_name)    
                        model.ds = ds
                        model.p = p
                        model.create_model()
                        
                        model.fit()
                        print("Training complete")
                        
                       
                        
                        if model_name == 'LSTM' or model_name == 'SVR' or model_name == 'KNN' or model_name == 'RF' or model_name == 'GRU' or model_name == 'HYBRID' or model_name == 'TCN':
                            if output> 0:
                                if model_name in ml_names:
                                    n = len(ds.X_test_array) - output
                                    y= len(ds.y_test_array) - output
                                    to_predict = ds.X_test_array[n:]
                                else:
                                    n = len(ds.X_test) - output
                                    y= len(ds.y_test) - output
                                    to_predict = ds.X_test[n:]
                            else:
                                to_predict = ds.X_test
                                n = 0
                                y = 0
                            train_mean = model.evaluate()
                            preds = model.predict(to_predict)
                            preds = np.array(preds).reshape(-1, 1)
                            np_preds =  ds.inverse_transform_predictions(preds = preds)
                            preds = ds.inverse_differenced_dataset(df=df, diff_vals= np_preds)
                            
                            ds.df = df
                            ds.dataset_creation(detrended=True)
                            if h > 0:
                                preds = preds[:-h]

                            if len(ds.target_name) <= 1:
                                labels = ds.y_test_array[(h+y):(len(preds)+h+y)].reshape(-1, 1)
                                train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
                            else:
                                labels = ds.y_test_array[h:len(preds)+h]
                                train_labels = ds.y_train_array[h:len(train_mean) + h]
                        elif model_name =='ARIMA' or model_name =='GARCH':
                            labels = ds.ts_test.pd_dataframe()
                            labels = np.array(labels.values).reshape(-1, 1)
                            labels = np.concatenate(labels, axis=0)
                            if output > 0:
                                    n = len(ds.ts_test) - output
                                    labels = labels[n:]
                            to_predict = labels
                           
                            preds = list() 
                            if model_name =='ARIMA':
                                yhat= model.predict(to_predict)
                                preds= yhat[0]
                            else:
                                yhat =model.temp_model.forecast(horizon = len(to_predict))
                                preds = yhat.mean.values[-1, :]

                            print(preds)
                            preds = np.array(preds).reshape(-1, 1)
                            ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain =ds.get_ts_data(df=df) 
                            np_preds =  ds.inverse_transform_predictions(preds = preds, X=ts_ttrain)
                            
                            preds = ds.inverse_differenced_dataset(df=df, diff_vals= np_preds)
                            ds.df = df
                            ds.dataset_creation(detrended=True)
                            labels = ds.ts_test.pd_dataframe()
                            labels = np.array(labels.values).reshape(-1, 1)
                            labels = np.concatenate(labels, axis=0)    
                            if output > 0:
                                    n = len(ds.ts_test) - output
                                    labels = labels[n:]
                            predictions.append(preds)
                            true_values.append(labels)
                           
                                


                        print("MSE", mean_squared_error(labels, preds))
                        print("MAE", mean_absolute_error(labels, preds))
                        print("MAPE", mean_absolute_percentage_error(labels, preds))
                        print("RMSE", np.sqrt(mean_squared_error(labels, preds)))
                        rmses.append(np.sqrt(mean_squared_error(labels, preds)))
                        mapes.append(mean_absolute_percentage_error(labels, preds))
                        mses.append(mean_squared_error(labels, preds))
                        maes.append(mean_absolute_error(labels, preds))
                        predictions.append(preds)
                        true_values.append(labels)

                    
                    

                    optimal_index = mapes.index(min(mapes))
                    avg_mae = Average(maes)
                    avg_mse = Average(mses)
                    avg_rmse = Average(rmses)
                    avg_mape = Average(mapes)
                    rmses.append(avg_rmse)
                    mapes.append(avg_mape)
                    mses.append(avg_mse)
                    maes.append(avg_mae)
                    print("__________AVERAGE______________")
                    print("MSE", avg_mse)
                    print("MAE", avg_mae)
                    print("MAPE", avg_mape)
                    print("RMSE", avg_rmse)
                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmse = rmses, mape = mapes, filename = experiment_name)
                    save_results.save_output_csv(predictions[optimal_index], true_values[optimal_index], 'avg' + res, model.name,
                                                    bivariate=len(ds.target_name) > 1)
def ensemble_test(models, clusters):
    maes, mses, rmses, mapes = [], [], [], []

    for c in clusters:
        f_name = 'close-' + c + '-w30-h0'
        df = save_results.save_ensemble_prediction_csv(models, c, 'close', filename = f_name)
        mae = mean_absolute_error(df['ACTUAL'], df['PREDICTED'])
        mse = mean_squared_error(df['ACTUAL'], df['PREDICTED'])
        rmse  = np.sqrt(mean_squared_error(df['ACTUAL'], df['PREDICTED']))
        mape = mean_absolute_percentage_error(df['ACTUAL'], df['PREDICTED'])
        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        mapes.append(mape)
        print(c.upper())
        print('MSE: ', mse)
        print('RMSE:' , rmse)
        print('MAE: ', mae)
        print('MAPE: ', mape)

    df_metrics = pd.DataFrame(columns = ['CRYPTO', 'MSE', 'RMSE', 'MAE', 'MAPE'])
    df_metrics['CRYPTO'] = clusters
    df_metrics['RMSE'] = rmses
    df_metrics['MSE'] = mses
    df_metrics['MAPE'] = mapes
    df_metrics['MAE'] = maes
    f_Name = 'res/Ensemble-Learning-DL-Metrics.csv'
    df_metrics.to_csv(f_Name)


"""
for l in dl_names:
    print(l)
    total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = l, scaling=['minmax'], output = 180)

for d in ml_names:  
    print(d)
    if d == 'RF': ITERATIONS =10
    else: ITERATIONS = 1
    total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], output=180)
for s in stats_names:
    print(s)
    total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = s, scaling=['minmax'])
"""

#tft_test(wins = [30], horizons = [0], resources = ['close'], clusters =['btc','eth','ltc','xrp','xmr'], model_name ='TFT', scaling=['minmax'])

#dataset_test(wins = [1], horizons = [0], resources = ['close'], clusters = ['ltc'])
#hyper_param_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc'], model_name = 'ARIMA', scaling=['minmax'])
#for d in ml_names: detrended_data_all_models_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'])
#dataset_binance_test(wins = [1], horizons = [0], resources = ['close'], clusters = ['btc'])
#dataset_binance_test_2(wins = [1], horizons = [0], resources = ['close'], clusters = ['ltc'])
#tft_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name ='TFT', scaling=['minmax'])
#clist = ['crypto_task_btc.csv', 'crypto_task_xmr.csv', 'crypto_task_xrp.csv', 'crypto_task_eth.csv', 'crypto_task_ltc.csv']
#clist = ['crypto_task_btc.csv',   'crypto_task_xmr.csv']
#tft_hyperparam_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc'], model_name ='TFT', scaling=['minmax'])
#tft_combined_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['ltc','xrp'], model_name ='TFT', scaling=['minmax'], pred_crypto='ltc')
#first_diff_dataset_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['ltc','xrp'], model_name ='TFT', scaling=['minmax'])
#first_diff_tft_test(wins = [30], horizons = [0], resources = ['close'], clusters =  ['btc','eth','ltc','xrp','xmr'], model_name ='TFT', scaling=['minmax'])
"""
for d in dl_names: 
    print(d)
    first_diff_total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], output=180)
for d in ml_names:  
    print(d)
    if d == 'RF': ITERATIONS =10
    else: ITERATIONS = 1
    first_diff_total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], output=180)
"""


#first_diff_total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'GARCH', scaling=['minmax'])
#first_diff_total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'ARIMA', scaling=['minmax'])