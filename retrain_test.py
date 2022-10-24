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
import statistics as stat


ITERATIONS = 10
apiKey = API_KEY
apiSecurity = API_SECURITY
dl_names = [ 'LSTM', 'GRU', 'TCN', 'HYBRID']
ml_names = ['RF', 'SVR', 'KNN']
stats_names = ['ARIMA']
def r_squared(true, predicted):
    y = np.array(true)
    y_hat = np.array(predicted)
    
    y_m = y.mean()

    ss_res = np.sum((y-y_hat)**2)
    ss_tot = np.sum((y-y_m)**2)
    
    return 1 - (ss_res/ss_tot)

def Average(lst):
    return sum(lst) / len(lst)
def get_average_metrics(mses, rmses, maes, mapes, r2 = False, evs = False, medAE = False, rmsle = False, msle = False):
    if r2 and evs and medAE and rmsle and msle:
        avg_r2 = Average(r2)
        avg_medae = Average(medAE)
        avg_evs = Average(evs)
        avg_rmsle = Average(rmsle)
        avg_msle = Average(msle)
        r2.append(avg_r2)
        evs.append(avg_evs)
        medAE.append(avg_medae)
        msle.append(avg_msle)
        rmsle.append(avg_rmsle)
    avg_mae = Average(maes)
    avg_mse = Average(mses)
    avg_rmse = Average(rmses)
    avg_mape = Average(mapes)
    rmses.append(avg_rmse)
    mapes.append(avg_mape)
    mses.append(avg_mse)
    maes.append(avg_mae)






def first_diff_total_test_retrain(wins, horizons, resources, clusters, model_name, scaling, retrain):  
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:                   
                    all_predictions, all_normalised_predictions = [], []
                    all_train, all_normalised_train, all_train_labels, all_normalised_train_labels = [], [], [], []
                    all_labels, all_normalised_labels = [], []
                    n_mses, n_maes, n_rmses, n_mapes = [], [], [], []
                    n_r2_scores, n_evs_scores, n_medaes, n_rmsles, n_msles = [], [], [], [], []
                    mses, maes, rmses, mapes = [], [], [], []
                    r2_scores, evs_scores, medaes, rmsles, msles = [], [], [], [], []
                    all_inversed_labels =[]
                    for i in range(ITERATIONS):
                        predictions, true_values =[], []
                        train, true_train = [], []
                        n_predictions, n_true_values = [], []
                        n_train, n_true_train = [], []
                        add_split_value = 0
                        inversed_labels =[]
                        
                        for r, val in enumerate(retrain):
                            output = val
                            experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                           
                            ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                        horizon=h, training_features=['close'],
                                                        target_name=['close'], train_split_factor=0.9, apiKey= apiKey, apiSecurity= apiSecurity)
                            df, diff_df = ds.differenced_dataset()
                            ds.df = diff_df
                            
                            if r > 0:
                                add_split_value += val
                            ds.add_split_value = add_split_value
                            ds.dataset_creation(detrended=True)
                            ds.dataset_normalization(scaling)
                            ds.data_summary()
                            parameters = pd.read_csv("hyperparams/p_hbnn-" + 'a' + ".csv").iloc[0]
                            
                            #files = sorted(glob.glob("saved_models/talos-HBNN-" + c + "-cpu-w" + str(win) + "-h" + str(h) + "*_weights.tf.i*"))

                            dense_act = 'relu'
                            if 'relu' in parameters['first_dense_activation']:
                                dense_act = 'relu'
                            elif 'tanh' in parameters['first_dense_activation']:
                                dense_act = 'tanh'
                            if r == 0:
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
                                        'first_lstm_dim': 75,
                                        'lstm_activation': 'relu',
                                        'first_dropout_rate': 0.05, 
                                        'second_lstm_dim':  50,
                                        'first_dense_dim':  32,
                                        'dense_activation': 'relu',
                                        'dense_kernel_init': 'he_normal',
                                        'gru_dim':50,
                                        'gru_activation':'relu',
                                        'second_dropout_rate':  0.0,
                                        'second_dense_dim':  64,
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
                                
                                model.p = p
                                model.ds = ds
                                model.create_model()
                            elif model_name in ml_names:
                                if model_name == 'SVR':
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
                                
                                if model_name == 'SVR':
                                    model = SVR(experiment_name)
                                elif model_name == 'RF':
                                    model = RF(experiment_name)
                                elif model_name == 'KNN':
                                    model = KNN(experiment_name)
                                   
                                
                                model.p = p
                                model.ds = ds
                                model.create_model()
                            elif model_name =='ARIMA':
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

                                print("RESOURCE:", res, "CLUSTER:", c, "HORIZON:", h, "WIN:", win)
                               
                                model = ARIMA(experiment_name) 
                                model.p = p
                                model.ds = ds
                                model.create_model()

                            else:
                                model.ds = ds
                                model.load_model()
                            model.fit()
                            print("Training complete")
                            

                            if model_name == 'LSTM' or model_name == 'SVR' or model_name == 'KNN' or model_name == 'RF' or model_name == 'GRU' or model_name == 'HYBRID' or model_name == 'TCN':
                                if output> 0:
                                    actual = ds.y_test_array[:output]
                                    if model_name in ml_names:
                                        to_predict = ds.X_test_array[:output]
                                       
                                    else:
                                        to_predict = ds.X_test[:output]
                                else:
                                    to_predict = ds.X_test
                                    actual = ds.y_test_array
                                
                                preds = model.predict(to_predict)
                                preds = np.array(preds).reshape(-1, 1)
                                train_mean = model.evaluate()
                                train_mean = np.array(train_mean).reshape(-1, 1)
                                np_actual = ds.inverse_transform_predictions(preds = actual)
                                np_preds = ds.inverse_transform_predictions(preds = preds)
                                np_train = ds.inverse_transform_predictions(preds = train_mean)
                                inversed_preds = ds.inverse_differenced_dataset(diff_vals= np_preds, df=df, l = len(ds.y_test_array))
                                inversed_actual = ds.inverse_differenced_dataset(diff_vals= np_actual, df=df, l = len(ds.y_test_array))
                                
                                inversed_train = ds.inverse_differenced_dataset(diff_vals= np_train, df=df, l = len(df))

                               
                                if h > 0:
                                    inversed_preds= inversed_preds[:-h]
                                ds.df = df
                                ds.dataset_creation(detrended=True)
                                if len(ds.target_name) <= 1:
                                    labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1)
                                    train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
                                else:
                                    labels = ds.y_test_array[h:len(inversed_preds)+h]
                                    train_labels = ds.y_train_array[h:len(train_mean) + h]
                                ds.dataset_normalization(scaling)
                                norm_preds = ds.scale_predictions(preds= inversed_preds)
                                norm_train =  ds.scale_predictions(preds= inversed_train)
                                if h > 0:
                                    norm_preds= norm_preds[:-h]

                                if len(ds.target_name) <= 1:
                                    n_labels = ds.y_test_array[(h):(len(norm_preds)+h)].reshape(-1, 1)
                                    n_train_labels = ds.y_train_array[h:len(train_mean)+h].reshape(-1, 1)
                                else:
                                    n_labels = ds.y_test_array[h:len(norm_preds)+h]
                                    n_train_labels = ds.y_train_array[h:len(train_mean) + h]
                            
                            elif model_name =='ARIMA' or model_name =='GARCH':
                                labels = ds.X_test_array
                                if output > 0:
                                    labels = labels[:output]
                                to_predict = labels
                               
                                
                                if model_name =='ARIMA':
                                    yhat, y_std = model.predict(to_predict)
                                    preds = yhat                                    
                                    train_mean, train_std = model.evaluate()
                                    
                                else:
                                    yhat = model.temp_model.forecast(horizon = len(to_predict))
                                    preds = yhat.mean.values[-1, :]
                                
                                
                                preds = np.array(preds).reshape(-1, 1)
                                train_mean = np.array(train_mean).reshape(-1, 1)
                                ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain = ds.get_ts_data(df=diff_df) 
                                np_preds =  ds.inverse_transform_predictions(preds = preds)
                                np_preds = np.array(np_preds).reshape(-1,1)
                               
                                np_train = ds.inverse_transform_predictions(preds = train_mean)
                                inversed_preds = ds.inverse_differenced_dataset(df=df, diff_vals= np_preds, l= (len(ds.y_test_array)))
                                inversed_train = ds.inverse_differenced_dataset(df=df, diff_vals= np_train, l = len(df))
                                ds.df = df
                                ds.dataset_creation(detrended=True)
                                if len(ds.target_name) <= 1:
                                    labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1)
                                    train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
                                else:
                                    labels = ds.y_test_array[h:len(inversed_preds)+h]
                                    train_labels = ds.y_train_array[h:len(train_mean) + h]

                               
                                ds.dataset_normalization(scaling)
                                ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain = ds.get_ts_data(df=df) 
                                norm_preds = ds.scale_predictions(preds= inversed_preds)
                                norm_train =  ds.scale_predictions(preds= inversed_train)
                                if len(ds.target_name) <= 1:
                                    n_labels = ds.y_test_array[(h):(len(norm_preds)+h)].reshape(-1, 1)
                                    n_train_labels = ds.y_train_array[h:len(train_mean)+h].reshape(-1, 1)
                                else:
                                    n_labels = ds.y_test_array[h:len(norm_preds)+h]
                                    n_train_labels = ds.y_train_array[h:len(train_mean) + h]
                                
                                    
                            
                            
                            predictions.extend(inversed_preds)
                            true_values.extend(labels)
                            train.extend(inversed_train)
                            inversed_labels.extend(inversed_actual)
                            true_train.extend(train_labels)
                            n_predictions.extend(norm_preds)
                            n_true_values.extend(n_labels)
                            n_train.extend(norm_train)
                            n_true_train.extend(n_train_labels)

                        #After Inverting Normalisation and First Difference
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
                        msles.append(mean_squared_log_error(true_values, predictions))
                        rmsles.append(np.sqrt(mean_squared_log_error(true_values, predictions)))
                        all_predictions.append(predictions)
                        all_labels.append(true_values)
                        all_train.append(train)
                        all_train_labels.append(true_train)
                        all_inversed_labels.append(inversed_labels)
                        #Scaling Outputs
                        print("MSE", mean_squared_error(n_true_values, n_predictions))
                        print("MAE", mean_absolute_error(n_true_values, n_predictions))
                        print("MAPE", mean_absolute_percentage_error(n_true_values, n_predictions))
                        print("RMSE", np.sqrt(mean_squared_error(n_true_values, n_predictions)))
                        n_rmses.append(np.sqrt(mean_squared_error(n_true_values, n_predictions)))
                        n_mapes.append(mean_absolute_percentage_error(n_true_values, n_predictions))
                        n_mses.append(mean_squared_error(n_true_values, n_predictions))
                        n_maes.append(mean_absolute_error(n_true_values, n_predictions))
                        n_r2_scores.append(r_squared(n_true_values, n_predictions))
                        n_medaes.append(median_absolute_error(n_true_values, n_predictions))
                        n_evs_scores.append(explained_variance_score(n_true_values, n_predictions))
                        n_msles.append(mean_squared_log_error(n_true_values, n_predictions))
                        n_rmsles.append(np.sqrt(mean_squared_log_error(n_true_values, n_predictions)))
                        all_normalised_predictions.append(n_predictions)
                        all_normalised_labels.append(n_true_values)
                        all_normalised_train.append(n_train)
                        all_normalised_train_labels.append(n_true_train)


                    optimal_index = mapes.index(min(mapes))
                    get_average_metrics(mses = mses, rmses =rmses, maes = maes, mapes = mapes, r2 = r2_scores, evs = evs_scores, medAE = medaes, rmsle = rmsles, msle = msles)
                    get_average_metrics(mses = n_mses, rmses =n_rmses, maes = n_maes, mapes = n_mapes,  r2 = n_r2_scores, evs = n_evs_scores, medAE = n_medaes, rmsle = n_rmsles, msle = n_msles)

                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores, evs = evs_scores, medAE = medaes, rmsle = rmsles, msle = msles)
                    save_results.save_output_csv(all_predictions[optimal_index], all_labels[optimal_index], res, model.name,
                                                    bivariate=len(ds.target_name) > 1)
                    save_results.save_output_csv(all_train[optimal_index], all_train_labels[optimal_index], res, 'train-' + model.name,
                                                        bivariate=len(ds.target_name) > 1)
                    #save_results.save_iteration_output_csv(preds= all_predictions, labels = all_inversed_labels, filename = model.name, iterations = ITERATIONS)
                    norm_file_name = model.name + '_N'
                    save_results.save_output_csv(all_normalised_predictions[optimal_index],all_normalised_labels[optimal_index], res, norm_file_name,
                                                    bivariate=len(ds.target_name) > 1)
                    save_results.save_output_csv(all_normalised_train[optimal_index], all_normalised_train_labels[optimal_index], res, 'train-' + norm_file_name,
                                                        bivariate=len(ds.target_name) > 1)
                    save_results.save_iteration_output_csv(preds= all_normalised_predictions, labels = all_normalised_labels, filename = norm_file_name, iterations = ITERATIONS)
                    norm_experiment_name = experiment_name + '-N'
                    save_results.save_metrics_csv(mses=n_mses, maes = n_maes,rmses = n_rmses, mapes = n_mapes, filename = norm_experiment_name, r2 = n_r2_scores, evs = n_evs_scores, medAE = n_medaes, rmsle = n_rmsles, msle = n_msles)          

def first_diff_tft_test_retrain(wins, horizons, resources, clusters, model_name, scaling,  retrain = False):
    for win in wins:
        for res in resources:
            for h in horizons:
                for c in clusters:
                    all_predictions, all_normalised_predictions = [], []
                    all_train, all_normalised_train, all_train_labels, all_normalised_train_labels = [], [], [], []
                    all_labels, all_normalised_labels = [], []
                    n_mses, n_maes, n_rmses, n_mapes = [], [], [], []
                    n_r2_scores, n_evs_scores, n_medaes, n_rmsles, n_msles = [], [], [], [], []
                    mses, maes, rmses, mapes = [], [], [], []
                    r2_scores, evs_scores, medaes, rmsles, msles = [], [], [], [], []
                    all_inversed_labels =[]
                    for i in range(ITERATIONS):
                        predictions, true_values =[], []
                        train, true_train = [], []
                        n_predictions, n_true_values = [], []
                        n_train, n_true_train = [], []
                        inversed_l = []
                        add_split_value = 0
                        for r, val in enumerate(retrain):
                            output = val
                            print(output)
                            experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)
                            ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                                        horizon=h, training_features=['close'],
                                                                        target_name=['close'], train_split_factor=0.9, apiKey= apiKey, apiSecurity= apiSecurity)
                            df, diff_df = ds.differenced_dataset()
                            ds.df = diff_df
                            if r > 0:
                                add_split_value += 30
                            ds.add_split_value = add_split_value          
                            ds.dataset_creation(detrended=True)
                            ds.dataset_normalization(scaling)  # , 'standard'])
                            ds.data_summary()
                            
                            if model_name =='TFT':
                                if r == 0:
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
                                else: 
                                    model.load_model()
                                model.ds = ds
                                model.fit(use_covariates = False)
                                
                                to_predict= ds.ts_test.pd_dataframe()
                                to_predict= np.array(to_predict.values).reshape(-1, 1)
                                if output > 0:
                                    to_predict = to_predict[:output]
                                ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain = ds.get_ts_data(df=diff_df)            
    
                                preds = model.predict(to_predict)
                                train_mean = model.evaluate()
                                
                                preds = preds.pd_dataframe()
                                train_mean = train_mean.pd_dataframe()
                                preds = np.array(preds.values).reshape(-1, 1)
                                train_mean = np.array(train_mean.values).reshape(-1, 1)
                                preds = ds.inverse_transform_predictions(preds= preds, method=scaling[0], X= ts_ttrain)
                                actual = ds.inverse_transform_predictions(preds = to_predict, method = scaling[0], X= ts_ttrain)
                                np_train = ds.inverse_transform_predictions(preds = train_mean, method = scaling[0], X=ts_ttrain)
                                inversed_preds= ds.inverse_differenced_dataset(df=df, diff_vals=preds, l= (len(ds.y_test_array)+1))
                                inversed_labels =ds.inverse_differenced_dataset(df= df, diff_vals=actual,  l= (len(ds.y_test_array)+1) )
                                ds.df = df
                                ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain = ds.get_ts_data(df=df)  
                                ds.dataset_creation(detrended=True)
                                labels = ts_test.pd_dataframe()
                                labels = np.array(labels.values).reshape(-1, 1)
                                if output > 0:
                                    labels = labels[:output]
                                
                                train_labels = ds.ts_ttrain.pd_dataframe()
                                train_labels = np.array(train_labels.values).reshape(-1, 1)
                                inversed_train = ds.inverse_differenced_dataset(df=df, diff_vals= np_train, l = len(df))
                                
                                if len(ds.target_name) <= 1:
                                    train_labels =  train_labels[h:len(inversed_train) + h].reshape(-1, 1)
                                else:
                                    train_labels =  train_labels[h:len(inversed_train) + h]
                                ds.dataset_normalization(scaling)
                                n_preds = ds.scale_predictions(preds= inversed_preds, method=scaling[0], X= ts_ttrain)
                                norm_train =  ds.scale_predictions(preds= inversed_train, method=scaling[0], X= ts_ttrain)
                                norm_train = norm_train[1:]
                                if len(ds.target_name) <= 1:
                                    n_labels = ds.y_test_array[(h):(len(n_preds)+h)].reshape(-1, 1)
                                    n_train_labels = ds.y_train_array[h:len(norm_train)+h].reshape(-1, 1)
                                else:
                                    n_labels = ds.y_test_array[h:len(n_preds)+h]
                                    n_train_labels = ds.y_train_array[h:len(norm_train) + h]
                            
                                
                                labels = labels[:-1]
                                inversed_preds = inversed_preds[1:]
                                n_preds = n_preds[1:]
                                n_labels = n_labels[:-1]
                                

                                predictions.extend(inversed_preds)
                                true_values.extend(labels)
                                inversed_l.extend(inversed_labels)
                                train.extend(inversed_train)
                                true_train.extend(train_labels)
                                n_predictions.extend(n_preds)
                                n_true_values.extend(n_labels)
                                n_train.extend(norm_train)
                                n_true_train.extend(n_train_labels)
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
                        msles.append(mean_squared_log_error(true_values, predictions))
                        rmsles.append(np.sqrt(mean_squared_log_error(true_values, predictions)))
                        all_predictions.append(predictions)
                        all_labels.append(true_values)
                        all_train.append(train)
                        all_train_labels.append(true_train)
                        all_inversed_labels.append(inversed_l)
                        #Scaling Outputs
                        print("MSE", mean_squared_error(n_true_values, n_predictions))
                        print("MAE", mean_absolute_error(n_true_values, n_predictions))
                        print("MAPE", mean_absolute_percentage_error(n_true_values, n_predictions))
                        print("RMSE", np.sqrt(mean_squared_error(n_true_values, n_predictions)))
                        n_rmses.append(np.sqrt(mean_squared_error(n_true_values, n_predictions)))
                        n_mapes.append(mean_absolute_percentage_error(n_true_values, n_predictions))
                        n_mses.append(mean_squared_error(n_true_values, n_predictions))
                        n_maes.append(mean_absolute_error(n_true_values, n_predictions))
                        n_r2_scores.append(r_squared(n_true_values, n_predictions))
                        n_medaes.append(median_absolute_error(n_true_values, n_predictions))
                        n_evs_scores.append(explained_variance_score(n_true_values, n_predictions))
                        n_msles.append(mean_squared_log_error(n_true_values, n_predictions))
                        n_rmsles.append(np.sqrt(mean_squared_log_error(n_true_values, n_predictions)))
                        all_normalised_predictions.append(n_predictions)
                        all_normalised_labels.append(n_true_values)
                        all_normalised_train.append(n_train)
                        all_normalised_train_labels.append(n_true_train)


                    optimal_index = mapes.index(min(mapes))
                    get_average_metrics(mses = mses, rmses =rmses, maes = maes, mapes = mapes, r2 = r2_scores, evs = evs_scores, medAE = medaes, rmsle = rmsles, msle = msles)
                    get_average_metrics(mses = n_mses, rmses =n_rmses, maes = n_maes, mapes = n_mapes,  r2 = n_r2_scores, evs = n_evs_scores, medAE = n_medaes, rmsle = n_rmsles, msle = n_msles)

                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores, evs = evs_scores, medAE = medaes, rmsle = rmsles, msle = msles)
                    save_results.save_output_csv(all_predictions[optimal_index], all_labels[optimal_index], res, model.name,
                                                    bivariate=len(ds.target_name) > 1)
                    save_results.save_output_csv(all_train[optimal_index], all_train_labels[optimal_index], res, 'train-' + model.name,
                                                        bivariate=len(ds.target_name) > 1)
                    save_results.save_iteration_output_csv(preds= all_predictions, labels = all_inversed_labels, filename = model.name, iterations = ITERATIONS)
                    norm_file_name = model.name + '_N'
                    save_results.save_output_csv(all_normalised_predictions[optimal_index],all_normalised_labels[optimal_index], res, norm_file_name,
                                                    bivariate=len(ds.target_name) > 1)
                    save_results.save_output_csv(all_normalised_train[optimal_index], all_normalised_train_labels[optimal_index], res, 'train-' + norm_file_name,
                                                        bivariate=len(ds.target_name) > 1)
                    save_results.save_iteration_output_csv(preds= all_normalised_predictions, labels = all_normalised_labels, filename = norm_file_name, iterations = ITERATIONS)
                    norm_experiment_name = experiment_name + '-N'
                    save_results.save_metrics_csv(mses=n_mses, maes = n_maes,rmses = n_rmses, mapes = n_mapes, filename = norm_experiment_name, r2 = n_r2_scores, evs = n_evs_scores, medAE = n_medaes, rmsle = n_rmsles, msle = n_msles)          


#all_models = ['HYBRID', 'LSTM', 'GRU', 'TCN', 'TFT', 'KNN', 'RF']

#first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['xmr'], model_name = 'HYBRID', scaling=['minmax'], retrain = [30])

#first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'ARIMA', scaling=['minmax'], retrain = [30])
"""
for d in  dl_names: 
    print(d)
    ITERATIONS = 1
    first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], retrain = [30])
"""
first_diff_tft_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'TFT', scaling=['minmax'], retrain = [31])
"""
for d in ml_names:  
    print(d)
    if d == 'RF': ITERATIONS = 10 
    else: ITERATIONS = 1
    first_diff_total_test_retrain(wins = [30], resources = ['close'], horizons = [0], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], retrain = [30])
"""

"""
for d in  dl_names: 
    print(d)
    first_diff_total_test_retrain_2(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], retrain = [30, 30, 30])
"""


