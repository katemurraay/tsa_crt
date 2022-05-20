import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from models.dl.hbnn import HBNN
from models.dl.lstm import LSTM
from models.dl.lstmd import LSTMD
from models.stats.arima import ARIMA
from models.stats.garch import GARCH
from models.model_probabilistic import ModelProbabilistic
from models.dl.model_probabilistic_dl import ModelProbabilisticDL
from models.dl.model_interface_dl import ModelInterfaceDL
from models.ml.svr import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from util import dataset, plot_training, save_results

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
wins = [288]
hs = [0]
resources = ['cpu']  # , 'mem']
clusters = ['a']  # , 'b', 'c', 'd', 'e', 'f', 'g', 'h']
model_name = 'LSTMD'
# if model_name in ['ARIMA', 'GARCH', 'SVR']:
#     wins = [1]

for win in wins:
    for res in resources:
        for h in hs:
            for c in clusters:
                mses, maes = [], []
                experiment_name = model_name + '-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                # Data creation and load
                # ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                #                               horizon=h, training_features=['avgcpu', 'time', 'avgmem'],
                #                               target_name=['avg' + res, 'avgmem'], train_split_factor=0.8)

                ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                                              horizon=h, training_features=['avgcpu'],
                                              target_name=['avg' + res], train_split_factor=0.8)

                ds.dataset_creation()
                ds.dataset_normalization(['standard'])  # , 'minmax', 'minmax'])

                # ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=1, output_window=1,
                #                               horizon=h, training_features=['avgcpu'],
                #                               target_name=['avgcpu'], train_split_factor=0.95)
                #
                # ds.dataset_creation()
                # ds.dataset_normalization(['minmax'])

                ds.data_summary()
                parameters = pd.read_csv("hyperparams/p_hbnn-" + c + ".csv").iloc[0]

                files = sorted(
                    glob.glob("saved_models/talos-HBNN-" + c + "-cpu-w" + str(win) + "-h" + str(h) + "*_weights.tf.i*"))

                dense_act = 'relu'
                if 'relu' in parameters['first_dense_activation']:
                    dense_act = 'relu'
                elif 'tanh' in parameters['first_dense_activation']:
                    dense_act = 'tanh'

                # HBNN
                if model_name == 'HBNN' or model_name == 'LSTM' or model_name == 'LSTMD':
                    p = {'first_conv_dim': parameters['first_conv_dim'],
                         'first_conv_activation': parameters['first_conv_activation'],
                         'first_conv_kernel': 7,
                         'first_lstm_dim': parameters['second_lstm_dim'],
                         'first_dense_dim': parameters['first_dense_dim'],
                         'first_dense_activation': dense_act,
                         'batch_size': parameters['batch_size'],
                         'epochs': 1,  # parameters['epochs'],
                         'patience': parameters['patience'],
                         'optimizer': parameters['optimizer'],
                         'lr': parameters['lr'],
                         'momentum': parameters['momentum'],
                         'decay': parameters['decay'],
                         'pred_steps': 0,
                         }

                # # ARIMA
                elif model_name == 'ARIMA':
                    p = {'p': 2,
                         'd': 0,
                         'q': 2,
                         'P': 0,
                         'Q': 0,
                         'D': 0,
                         'S': 12,
                         'loop': 0,
                         'horizon': 2,
                         }
                elif model_name == 'GARCH':
                    # # GARCH
                    p = {'p': 1,
                         'q': 1,
                         'loop': 0,
                         'horizon': 2,
                         'mean': 'LS',
                         }
                elif model_name == 'SVR':
                    # # SVR
                    p = {'kernel': 'rbf',
                         'degree': 3,
                         'tol': 0.001,
                         'C': 1.0,
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

                model.ds = ds
                model.p = p
                model.create_model()

                model.fit()
                print("Training complete")
                # model.hyperparametrization()
                if model_name == 'LSTM' or model_name == 'SVR' or model_name == 'KNN' or model_name == 'RF':
                    train_mean = model.evaluate()
                    preds = model.predict(ds.X_test)

                    if h > 0:
                        preds = preds[:-h]

                    if len(ds.target_name) <= 1:
                        labels = ds.y_test_array[h:len(preds) + h].reshape(-1, 1)
                        train_labels = ds.y_train_array[h:len(train_mean) + h].reshape(-1, 1)
                    else:
                        labels = ds.y_test_array[h:len(preds) + h]
                        train_labels = ds.y_train_array[h:len(train_mean) + h]

                    print("MSE", mean_squared_error(labels, preds))
                    print("MAE", mean_absolute_error(labels, preds))

                    plot_training.plot_series(np.arange(0, len(preds)), labels, preds, label1="ground truth",
                                              label2="prediction", title=model.name, bivariate=len(ds.target_name) > 1)

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

                    save_results.save_params_csv(p, model.name)

                else:

                    train_mean, train_std = model.evaluate()
                    prediction_mean, prediction_std = model.predict(ds.X_test)

                    if len(ds.target_name) <= 1:
                        a = np.concatenate(ds.y_train[:len(train_mean)], axis=0).reshape(-1, 1)
                        b = np.concatenate(ds.y_test[:len(prediction_mean)], axis=0).reshape(-1, 1)
                        if isinstance(model, ModelProbabilistic) and not isinstance(model, ModelProbabilisticDL):
                            a = ds.y_train_array
                            b = ds.y_test_array
                    else:
                        a = ds.y_train[:len(train_mean)]
                        b = ds.y_test[:len(prediction_mean)]
                    save_results.save_uncertainty_csv(train_mean, train_std, a,
                                                      'avg' + res,
                                                      'train-' + model.name, bivariate=len(ds.target_name) > 1)
                    save_results.save_uncertainty_csv(prediction_mean, prediction_std, b,
                                                      'avg' + res,
                                                      model.name, bivariate=len(ds.target_name) > 1)

                    plot_training.plot_series_interval(np.arange(0, len(ds.y_test) - 1), ds.y_test, prediction_mean,
                                                       prediction_std,
                                                       label1="ground truth",
                                                       label2="prediction", title=model.name,
                                                       bivariate=len(ds.target_name) > 1)
                    if model_name in ['LSTM', 'LSTMD', 'HBNN']:
                        plot_model(model.model, to_file='img/models/model_plot_' + model.name + '.png',
                                   show_shapes=True,
                                   show_layer_names=True)
