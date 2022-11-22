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

from keras import backend as K

ITERATIONS = 3
apiKey = API_KEY
apiSecurity = API_SECURITY
dl_names= [ 'LSTM', 'GRU', 'HYBRID']
#[ 'TCN',  'LSTM', 'HYBRID'] 
ml_names = ['SVR', 'KNN', 'RF']
stats_names = ['ARIMA']
thirty_day_months = [9 , 11, 4, 6]
def r_squared(true, predicted):
    y = np.array(true)
    y_hat = np.array(predicted)
    y_m = y.mean()
    ss_res = np.sum((y-y_hat)**2)
    ss_tot = np.sum((y-y_m)**2)
    
    return 1 - (ss_res/ss_tot)

def Average(lst):
    return sum(lst) / len(lst)
def get_average_metrics(mses, rmses, maes, mapes, r2 = False):
    avg_r2 = Average(r2)
    r2.append(avg_r2)
    avg_mae = Average(maes)
    avg_mse = Average(mses)
    avg_rmse = Average(rmses)
    avg_mape = Average(mapes)
    rmses.append(avg_rmse)
    mapes.append(avg_mape)
    mses.append(avg_mse)
    maes.append(avg_mae)


def first_diff_total_test_retrain(wins, horizons, resources, clusters, model_name, scaling, retrain, outputs):  
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
                    all_inversed_labels, all_training_time, all_inference_time =[], [], []
                    for i in range(ITERATIONS):
                        predictions, true_values =[], []
                        train, true_train = [], []
                        n_predictions, n_true_values = [], []
                        n_train, n_true_train = [], []
                        add_split_value = 0
                        inversed_labels =[]
                        training_time, inference_time = [], []
                        w_mses, w_maes, w_rmses, w_mapes, w_r2 = [], [], [], [], []
                        w_nmses, w_nmaes, w_nrmses, w_nmapes, w_nr2 = [], [], [], [], []
                        w_predictions, w_labels, w_npredictions, w_nlabels = [], [], [], []
                        for r, month in enumerate(retrain):                           
                            val = 31
                            if month in thirty_day_months: val = 30
                            elif month == 2: val = 28

                            if outputs[r] in thirty_day_months: output = 30
                            elif outputs[r] == 2: output = 28
                            else: output = 31
                            

                            #experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h) 
                            experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h) + '-' + str(len(retrain)) + 'm'
                            ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                        horizon=h, training_features=['close'],
                                                        target_name=['close'], train_split_factor=0.8, apiKey= apiKey, apiSecurity= apiSecurity)
                            df, diff_df = ds.differenced_dataset()
                            ds.df = diff_df
                           
                            if r > 0:
                                add_split_value += val
                                ds.add_split_value = add_split_value
                            else:
                                ds.add_split_value = 0
                            
                            ds.dataset_creation(df=True, detrended= True)
                            ds.dataset_normalization(scaling)
                            ds.data_summary()
                            parameters = pd.read_csv("hyperparams/p_hbnn-" + 'a' + ".csv").iloc[0]
                            
                            #files = sorted(glob.glob("saved_models/talos-HBNN-" + c + "-cpu-w" + str(win) + "-h" + str(h) + "*_weights.tf.i*"))

                            dense_act = 'relu'
                            if 'relu' in parameters['first_dense_activation']:
                                dense_act = 'relu'
                            elif 'tanh' in parameters['first_dense_activation']:
                                dense_act = 'tanh'
                        
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
                            # ARIMA
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
                                # GARCH
                                p = {'p': 2,
                                    'q': 1,
                                    'loop': 0,
                                    'horizon': 2,
                                    'mean': 'LS',
                                    }
                            elif model_name == 'SVR':
                                # SVR
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

                            if r == 0: 
                                K.clear_session()
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
                            torch.cuda.empty_cache()
                            device = torch.device("cuda")
                            
                            #dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(device)

                            # INIT LOGGERS
                            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            repetitions = 1
                            timings=np.zeros((repetitions,1))
                            
                            for _ in range(10):
                                model.ds = ds
                            
                                if r == 0: 
                                    model.p = p
                                    model.create_model()
                                elif model_name in ml_names:
                                    model.p = p
                                    model.create_model()
                                else: model.load_model()
                            # MEASURE PERFORMANCE
                            with torch.no_grad():
                                for rep in range(repetitions):
                                    starter.record()
                                    model.fit()
                                    ender.record()
                                    torch.cuda.synchronize()
                                    curr_time = starter.elapsed_time(ender)
                                    timings[rep] = curr_time

                            train_time = np.sum(timings) / repetitions
                            std_syn = np.std(timings)
                            training_time.append(train_time)
                            
                               
                            
                            print("Training complete")
                            

                            if model_name == 'LSTM' or model_name == 'SVR' or model_name == 'KNN' or model_name == 'RF' or model_name == 'GRU' or model_name == 'HYBRID' or model_name == 'TCN':
                                
                                

                                # INIT LOGGERS
                                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                repetitions = 1
                                timings=np.zeros((repetitions,1))
                                
                                for _ in range(10):
                                    if output> 0:
                                        actual = ds.y_test_array[:output]
                                        if model_name in ml_names:
                                            to_predict = ds.X_test_array[:output]
                                        
                                        else:
                                            to_predict = ds.X_test[:output]
                                    else:
                                        to_predict = ds.X_test
                                        actual = ds.y_test_array
                                # MEASURE PERFORMANCE
                                with torch.no_grad():
                                    for rep in range(repetitions):
                                        starter.record()
                                        preds = model.predict(to_predict)
                                        ender.record()
                                        torch.cuda.synchronize()
                                        curr_time = starter.elapsed_time(ender)
                                        timings[rep] = curr_time

                                inf_time = np.sum(timings) / len(to_predict)
                                std_syn = np.std(timings)
                                inference_time.append(inf_time)
                               
                                
                                
                                preds = np.array(preds).reshape(-1, 1)
                                train_mean = model.evaluate()
                                train_mean = np.array(train_mean).reshape(-1, 1)
                                np_actual = ds.inverse_transform_predictions(preds = actual)
                                np_preds = ds.inverse_transform_predictions(preds = preds)
                                np_train = ds.inverse_transform_predictions(preds = train_mean)
                                inversed_preds = ds.inverse_differenced_dataset(diff_vals= np_preds, df=df, l = (len(ds.y_test_array)))
                                inversed_actual = ds.inverse_differenced_dataset(diff_vals= np_actual, df=df, l = (len(ds.y_test_array)))             
                                inversed_train = ds.inverse_differenced_dataset(diff_vals= np_train, df=df, l = len(df))

                               
                                if h > 0:
                                    inversed_preds= inversed_preds[:-h]
                                                                
                                if r ==11:
                                    if model_name in ml_names:
                                        inversed_preds = inversed_preds[:-5]
                                ds.df = df
                                
                                ds.dataset_creation(df=True)
                               
                                if len(ds.target_name) <= 1:
                                    labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1)
                                    train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
                                else:
                                    labels = ds.y_test_array[h:len(inversed_preds)+h]
                                    train_labels = ds.y_train_array[h:len(train_mean) + h]
                                ds.df = df
                                if r > 0:  
                                    ds.add_split_value = 0
                                
                                ds.dataset_creation(df=True)
                                ds.dataset_normalization(scaling)
                                norm_preds = ds.scale_predictions(preds= inversed_preds)                               
                                n_labels =  ds.scale_predictions(preds= labels)


                            
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
                                np_preds =  ds.inverse_transform_predictions(preds = preds)
                                np_preds = np.array(np_preds).reshape(-1,1)
                                np_actual = np.array(ds.y_test_array).reshape(-1,1)
                                np_train = ds.inverse_transform_predictions(preds = train_mean)
                                inversed_preds = ds.inverse_differenced_dataset(df=df, diff_vals= np_preds, l= (len(ds.y_test_array)))
                                inversed_actual = ds.inverse_differenced_dataset(diff_vals= np_actual, df=df, l =len(ds.y_test_array))
                                inversed_train = ds.inverse_differenced_dataset(df=df, diff_vals= np_train, l = len(df))
                                ds.df = df
                                ds.dataset_creation(df=True)
                                if len(ds.target_name) <= 1:
                                    labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1)
                                    train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
                                else:
                                    labels = ds.y_test_array[h:len(inversed_preds)+h]
                                    train_labels = ds.y_train_array[h:len(train_mean) + h]

                                ds.add_split_value = 0
                                ds.df = df
                                ds.dataset_creation(df=True)
                                ds.dataset_normalization(scaling)
                                norm_preds = ds.scale_predictions(preds= inversed_preds)                               
                                n_labels =  ds.scale_predictions(preds= labels)
                             
                                
                                    
                            
                            
                            predictions.extend(inversed_preds)
                            true_values.extend(labels)
                            train.extend(inversed_train)
                            inversed_labels.extend(inversed_actual)
                            true_train.extend(train_labels)
                            n_predictions.extend(norm_preds)
                            n_true_values.extend(n_labels)
                        
                            w_predictions.append(inversed_preds)
                            w_npredictions.append(norm_preds)
                            w_labels.append(labels)
                            w_nlabels.append(n_labels)
                            w_rmses.append(np.sqrt(mean_squared_error(labels, inversed_preds)))
                            w_mapes.append(mean_absolute_percentage_error(labels, inversed_preds))
                            w_mses.append(mean_squared_error(labels, inversed_preds))
                            w_maes.append(mean_absolute_error(labels, inversed_preds))
                            w_r2.append(r_squared(labels, inversed_preds))
                            w_nrmses.append(np.sqrt(mean_squared_error(n_labels, norm_preds)))
                            w_nmapes.append(mean_absolute_percentage_error(n_labels, norm_preds))
                            w_nmses.append(mean_squared_error(n_labels, norm_preds))
                            w_nmaes.append(mean_absolute_error(n_labels, norm_preds))
                            w_nr2.append(r_squared(n_labels, norm_preds))
                        metric_name  = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h) + '-windowed-' + str(len(retrain)) + 'm-' + str(i)
                        save_results.save_metrics_csv(mses=w_mses, maes =  w_maes,rmses = w_rmses, mapes = w_mapes, filename = metric_name, r2 = w_r2, iterations=True)
                        n_metric_name = metric_name + '_N'
                        save_results.save_metrics_csv(mses=w_nmses, maes =  w_nmaes,rmses = w_nrmses, mapes = w_nmapes, filename = n_metric_name, r2 = w_nr2,  iterations=True)
                        save_results.save_window_outputs(labels = w_labels, preds = w_predictions, filename = metric_name)
                        save_results.save_window_outputs(labels = w_nlabels, preds = w_npredictions, filename = n_metric_name)
                        
                        
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
                        #msles.append(mean_squared_log_error(true_values, predictions))
                        #rmsles.append(np.sqrt(mean_squared_log_error(true_values, predictions)))
                        all_predictions.append(predictions)
                        all_labels.append(true_values)
                        all_train.append(train)
                        all_train_labels.append(true_train)
                        all_inversed_labels.append(inversed_labels)
                        all_training_time.append(training_time)
                        all_inference_time.append(inference_time)
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
                        #n_msles.append(mean_squared_log_error(n_true_values, n_predictions))
                        #n_rmsles.append(np.sqrt(mean_squared_log_error(n_true_values, n_predictions)))
                        all_normalised_predictions.append(n_predictions)
                        all_normalised_labels.append(n_true_values)
                        


                    optimal_index = -1
                    get_average_metrics(mses = mses, rmses =rmses, maes = maes, mapes = mapes, r2 = r2_scores)
                    get_average_metrics(mses = n_mses, rmses =n_rmses, maes = n_maes, mapes = n_mapes,  r2 = n_r2_scores)
                    
                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores)

                    save_results.save_output_csv(all_train[optimal_index], all_train_labels[optimal_index], res, 'train-' + experiment_name,
                                                        bivariate=len(ds.target_name) > 1)
                    save_results.save_iteration_output_csv(preds= all_predictions, labels = all_inversed_labels, filename = experiment_name, iterations = ITERATIONS)
                    inference_name = experiment_name + '-inf_time'
                    save_results.save_timing(times = all_inference_time, filename = inference_name, iterations = ITERATIONS)
                    training_name =  experiment_name + '-training_time'
                    save_results.save_timing(times = all_training_time, filename = training_name, iterations = ITERATIONS)
                    norm_file_name = experiment_name + '_N'
                    save_results.save_output_csv(all_normalised_predictions[optimal_index],all_normalised_labels[optimal_index], res, norm_file_name,
                                                    bivariate=len(ds.target_name) > 1)
                    
                    save_results.save_iteration_output_csv(preds= all_normalised_predictions, labels = all_normalised_labels, filename = norm_file_name, iterations = ITERATIONS)
                    norm_experiment_name = model.name + '_N'
                    save_results.save_metrics_csv(mses=n_mses, maes = n_maes,rmses = n_rmses, mapes = n_mapes, filename = norm_experiment_name, r2 = n_r2_scores)          
                
def first_diff_tft_test_retrain(wins, horizons, resources, clusters, model_name, scaling,  retrain = False, outputs = False):
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
                    all_inversed_labels, all_training_time, all_inference_time =[], [], []
                    for i in range(ITERATIONS):
                        predictions, true_values =[], []
                        train, true_train = [], []
                        n_predictions, n_true_values = [], []
                        n_train, n_true_train = [], []
                        inversed_l = []
                        add_split_value = 0
                        training_time, inference_time = [], []
                        w_mses, w_maes, w_rmses, w_mapes, w_r2 = [], [], [], [], []
                        w_nmses, w_nmaes, w_nrmses, w_nmapes, w_nr2 = [], [], [], [], []
                        w_predictions, w_labels, w_npredictions, w_nlabels = [], [], [], []
                        for r, month in enumerate(retrain):
                            val = 31
                            if month in thirty_day_months: val = 30
                            elif month == 2: val = 28

                            if outputs[r] in thirty_day_months: output = 30
                            elif outputs[r] == 2: output = 28
                            else: output = 31
                            print(output)
                            #experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h) 
                            experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)  + '-' + str(len(retrain)) + 'm'
                            ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=win, output_window=1,
                                                                        horizon=h, training_features=['close'],
                                                                        target_name=['close'], train_split_factor=0.8, apiKey= apiKey, apiSecurity= apiSecurity)
                            df, diff_df = ds.differenced_dataset()
                            ds.df = diff_df
                           
                            if r > 0:
                                add_split_value += val
                                ds.add_split_value = add_split_value
                            else:
                                ds.add_split_value = 0
                                
                                   
                            ds.dataset_creation(df = True, detrended=True)
                            ds.dataset_normalization(scaling)  # , 'standard'])
                            ds.data_summary()
                            if r ==0:
                                if model_name =='TFT':
                                    torch.cuda.empty_cache()
                                    model = TFT(experiment_name)
                                    p = {
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
                               
                                
                            # INIT LOGGERS
                            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            repetitions = 1
                            timings=np.zeros((repetitions,1))
                            
                            for _ in range(10):
                                model.ds = ds
                               
                                if r == 0: 
                                    model.p
                                    model.create_model()
                                else:
                                    model.load_model()
                            # MEASURE PERFORMANCE
                            with torch.no_grad():
                                for rep in range(repetitions):
                                    starter.record()
                                    model.fit(use_covariates = False)  
                                    ender.record()
                                    torch.cuda.synchronize()
                                    curr_time = starter.elapsed_time(ender)
                                    timings[rep] = curr_time

                            train_time = np.sum(timings) / repetitions
                            std_syn = np.std(timings)
                            training_time.append(train_time)
                            
                            
                            
                            print("Training complete")

                            
                                
                            
                            
                            
                            
                            ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain = ds.get_ts_data(df=diff_df)    
                            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            repetitions = 1
                            timings=np.zeros((repetitions,1))
                            for _ in range(10):
                                to_predict= ds.ts_test.pd_dataframe()
                                to_predict= np.array(to_predict.values).reshape(-1, 1)
                                if output> 0:
                                    to_predict = to_predict[:output]

                            # MEASURE PERFORMANCE
                            with torch.no_grad():
                                for rep in range(repetitions):
                                    starter.record()
                                    preds = model.predict(to_predict)
                                    ender.record()
                                    torch.cuda.synchronize()
                                    curr_time = starter.elapsed_time(ender)
                                    timings[rep] = curr_time

                            inf_time = np.sum(timings) / len(to_predict)
                            std_syn = np.std(timings)
                            inference_time.append(inf_time)
                            preds = model.predict(to_predict)
                            train_mean = model.evaluate()
                            
                            preds = preds.pd_dataframe()
                            train_mean = train_mean.pd_dataframe()
                            preds = np.array(preds.values).reshape(-1, 1)
                            train_mean = np.array(train_mean.values).reshape(-1, 1)
                            preds = ds.inverse_transform_predictions(preds= preds, method=scaling[0], X= ts_ttrain)
                            actual = ds.inverse_transform_predictions(preds = to_predict, method = scaling[0], X= ts_ttrain)
                            np_train = ds.inverse_transform_predictions(preds = train_mean, method = scaling[0], X=ts_ttrain)
                            inversed_preds= ds.inverse_differenced_dataset(df=df, diff_vals=preds, l= (len(ds.y_test_array)))
                            inversed_labels =ds.inverse_differenced_dataset(df= df, diff_vals=actual,  l= (len(ds.y_test_array)))
                            if r ==11:                               
                                inversed_preds = inversed_preds[:-5]
                            
                            ds.df = df
                            ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain = ds.get_ts_data(df=df)  
                            ds.dataset_creation(df = True)
                            labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1)
                            
                            
                            train_labels = ds.ts_ttrain.pd_dataframe()
                            train_labels = np.array(train_labels.values).reshape(-1, 1)
                            inversed_train = ds.inverse_differenced_dataset(df=df, diff_vals= np_train, l = len(df))
                            
                            if len(ds.target_name) <= 1:
                                train_labels =  train_labels[h:len(inversed_train) + h].reshape(-1, 1)
                            else:
                                train_labels =  train_labels[h:len(inversed_train) + h]
                            
                            ds.df = df
                            if r > 0: ds.add_split_value = 0 
                            ts_train, ts_val, ts_test, strain_cov, cov, ts_ttrain = ds.get_ts_data(df=df) 
                            
                            ds.dataset_creation(df = True)
                            ds.dataset_normalization(scaling)
                            n_preds = ds.scale_predictions(preds= inversed_preds, method=scaling[0], X= ts_ttrain)
                            n_labels =  ds.scale_predictions(preds= labels, method=scaling[0], X= ts_ttrain)
                            norm_train =  ds.scale_predictions(preds= inversed_train, method=scaling[0], X= ts_ttrain)
                            n_train_labels =  ds.scale_predictions(preds= train_labels, method=scaling[0], X= ts_ttrain)
                            
                            norm_train = norm_train[1:]

                            predictions.extend(inversed_preds)
                            true_values.extend(labels)
                            inversed_l.extend(inversed_labels)
                            train.extend(inversed_train)
                            true_train.extend(train_labels)
                            n_predictions.extend(n_preds)
                            n_true_values.extend(n_labels)
                            n_train.extend(norm_train)
                            n_true_train.extend(n_train_labels)
                            w_predictions.append(inversed_preds)
                            w_npredictions.append(n_preds)
                            w_labels.append(labels)
                            w_nlabels.append(n_labels)
                            w_rmses.append(np.sqrt(mean_squared_error(labels, inversed_preds)))
                            w_mapes.append(mean_absolute_percentage_error(labels, inversed_preds))
                            w_mses.append(mean_squared_error(labels, inversed_preds))
                            w_maes.append(mean_absolute_error(labels, inversed_preds))
                            w_r2.append(r_squared(labels, inversed_preds))
                            w_nrmses.append(np.sqrt(mean_squared_error(n_labels, n_preds)))
                            w_nmapes.append(mean_absolute_percentage_error(n_labels, n_preds))
                            w_nmses.append(mean_squared_error(n_labels, n_preds))
                            w_nmaes.append(mean_absolute_error(n_labels, n_preds))
                            w_nr2.append(r_squared(n_labels, n_preds))
                        metric_name  = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h) + '-windowed-' + str(len(retrain)) + 'm-' + str(i)
                        save_results.save_metrics_csv(mses=w_mses, maes =  w_maes,rmses = w_rmses, mapes = w_mapes, filename = metric_name, r2 = w_r2, iterations=True)
                        n_metric_name = metric_name + '_N'
                        save_results.save_metrics_csv(mses=w_nmses, maes =  w_nmaes,rmses = w_nrmses, mapes = w_nmapes, filename = n_metric_name, r2 = w_nr2,  iterations=True)
                        save_results.save_window_outputs(labels = w_labels, preds = w_predictions, filename = metric_name)
                        save_results.save_window_outputs(labels = w_nlabels, preds = w_npredictions, filename = n_metric_name)
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
                        all_training_time.append(training_time)
                        all_inference_time.append(inference_time)
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


                    optimal_index = -1
                    get_average_metrics(mses = mses, rmses =rmses, maes = maes, mapes = mapes, r2 = r2_scores)
                    get_average_metrics(mses = n_mses, rmses =n_rmses, maes = n_maes, mapes = n_mapes,  r2 = n_r2_scores)

                    save_results.save_metrics_csv(mses=mses, maes =  maes,rmses = rmses, mapes = mapes, filename = experiment_name, r2 = r2_scores)
                    save_results.save_output_csv(all_predictions[optimal_index], all_labels[optimal_index], res, model.name,
                                                    bivariate=len(ds.target_name) > 1)
                    save_results.save_output_csv(all_train[optimal_index], all_train_labels[optimal_index], res, 'train-' + model.name,
                                                        bivariate=len(ds.target_name) > 1)
                    save_results.save_iteration_output_csv(preds= all_predictions, labels = all_inversed_labels, filename = model.name, iterations = ITERATIONS)
                    norm_file_name = model.name + '_N'
                    save_results.save_output_csv(all_normalised_predictions[optimal_index],all_normalised_labels[optimal_index], res, norm_file_name,
                                                    bivariate=len(ds.target_name) > 1)
                    #save_results.save_output_csv(all_normalised_train[optimal_index], all_normalised_train_labels[optimal_index], res, 'train-' + norm_file_name,
                                                       # bivariate=len(ds.target_name) > 1)
                    inference_name = experiment_name + '-inf_time'
                    save_results.save_timing(times = all_inference_time, filename = inference_name, iterations = ITERATIONS)
                    training_name =  experiment_name + '-training_time'
                    save_results.save_timing(times = all_training_time, filename = training_name, iterations = ITERATIONS)
                  
                    save_results.save_iteration_output_csv(preds= all_normalised_predictions, labels = all_normalised_labels, filename = norm_file_name, iterations = ITERATIONS)
                    norm_experiment_name = experiment_name + '-N'
                    save_results.save_metrics_csv(mses=n_mses, maes = n_maes,rmses = n_rmses, mapes = n_mapes, filename = norm_experiment_name, r2 = n_r2_scores)          


 
dct_lstm_time = {'BTC': 14361.8095703125, 'ETH': 10038.1572265625, 'LTC': 11263.4453125, 'XRP': 9896.107421875, 'XMR': 13423.7294921875}
dct_gru_time = {'BTC': 77818.4453125, 'ETH': 54792.3125, 'LTC': 62416.484375, 'XRP': 55065.66015625, 'XMR': 77253.6484375}
dct_hybrid_time = {'BTC': 179469.140625, 'ETH': 127679.9140625, 'LTC': 145463.765625, 'XRP': 124436.3828125, 'XMR': 179966.3125}

#first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'LSTM', scaling=['minmax'], retrain = [30])

tft_timeframes = [ [31], [31, 31], [31, 31, 31]]
dl_timeframes = [[30, 30, 30], [30], [30, 30]]
all_models = ['HYBRID', 'LSTM', 'GRU', 'TCN', 'TFT', 'RF']
one_iteration_mod = ['ARIMA', 'SVR', 'KNN']
#start_datsets(clusters = ['btc','eth','ltc','xrp','xmr'])
"""
for m in all_models:
    
    fix_normalisation(models =m , timeframe = '3m', clusters=['btc','eth','ltc','xrp','xmr'], resources =['close'], scaling =['minmax'])
#
"""
#
#first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'ARIMA', scaling=['minmax'], retrain = [30])
"""
for d in  dl_names: 
    print(d)
    first_diff_total_test(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], output= 0)

for d in  dl_names: 
    print(d)
    
    first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], retrain = [0, 6, 7, 8, 9 , 10, 11, 12, 1, 2, 3, 4], outputs =[6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5])

"""
"""
ITERATIONS=1
first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'ARIMA', scaling=['minmax'], retrain = [0])
#first_diff_tft_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'TFT', scaling=['minmax'], retrain = [0])


for t in dl_timeframes:
    str_months = 'MONTHS: ' + str(len(t))
    print(str_months)
    for d in  dl_names: 
        print(d)
        first_diff_total_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], retrain = t)

for t in tft_timeframes:
    str_months = 'MONTHS: ' + str(len(t))
    print(str_months)
    
    first_diff_tft_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'TFT', scaling=['minmax'], retrain = t)

for d in ml_names:  
    print(d)
    if d == 'RF': ITERATIONS = 3
    else: ITERATIONS = 1
    first_diff_total_test_retrain(wins = [30], resources = ['close'], horizons = [0], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = d, scaling=['minmax'], retrain = [0, 6, 7, 8, 9 , 10, 11, 12, 1, 2, 3, 4], outputs =[6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5])
"""

first_diff_tft_test_retrain(wins = [30], horizons = [0], resources = ['close'], clusters = ['btc','eth','ltc','xrp','xmr'], model_name = 'TFT', scaling=['minmax'], retrain = [0, 6, 7, 8, 9 , 10, 11, 12, 1, 2, 3, 4], outputs =[6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5])

