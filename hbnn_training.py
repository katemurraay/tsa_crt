import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from models import HBNN
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from util import dataset, plot_training, save_results

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
wins = [288]
hs = [0]
resources = ['cpu']  # , 'mem']
clusters = ['a']  # , 'b', 'c', 'd', 'e', 'f', 'g', 'h']

for win in wins:
    for res in resources:
        for h in hs:
            for c in clusters:
                mses, maes = [], []
                experiment_name = 'HBNN-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                # Data creation and load
                # ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                #                               horizon=h, training_features=['avgcpu', 'time', 'avgmem'],
                #                               target_name=['avg' + res, 'avgmem'], train_split_factor=0.8)

                ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                                              horizon=h, training_features=['avgcpu', 'time', 'avgmem'],
                                              target_name=['avg' + res, 'avgmem'], train_split_factor=0.8)

                ds.dataset_creation()
                ds.dataset_normalization(['standard', 'minmax', 'minmax'])

                ds.data_summary()
                parameters = pd.read_csv("hyperparams/p_hbnn-" + c + ".csv").iloc[0]

                files = sorted(
                    glob.glob("saved_models/talos-HBNN-" + c + "-cpu-w" + str(win) + "-h" + str(h) + "*_weights.tf.i*"))

                dense_act = 'relu'
                if 'relu' in parameters['first_dense_activation']:
                    dense_act = 'relu'
                elif 'tanh' in parameters['first_dense_activation']:
                    dense_act = 'tanh'

                p = {'first_conv_dim': parameters['first_conv_dim'],
                     'first_conv_activation': parameters['first_conv_activation'],
                     # 'first_conv_kernel': (parameters['first_conv_kernel'],),
                     'second_lstm_dim': parameters['second_lstm_dim'],
                     'first_dense_dim': parameters['first_dense_dim'],
                     'first_dense_activation': dense_act,
                     'batch_size': parameters['batch_size'],
                     'epochs': parameters['epochs'],
                     'patience': parameters['patience'],
                     'optimizer': parameters['optimizer'],
                     'batch_normalization': True,
                     'lr': parameters['lr'],
                     'momentum': parameters['momentum'],
                     'decay': parameters['decay'],
                     'pred_steps': 0,
                     }

                print("RESOURCE:", res, "CLUSTER:", c, "HORIZON:", h, "WIN:", win)
                model = HBNN.HBNNPredictor()
                model.name = experiment_name

                if len(files):
                    for i in range(len(files)):
                        path_weight = files[-(i + 1)][:-6]
                        p['weight_file'] = path_weight
                        try:
                            train_model, prediction_mean, prediction_std = model.load_and_predict(ds.X_train,
                                                                                                  ds.y_train,
                                                                                                  ds.X_test,
                                                                                                  ds.y_test, p)
                        except:
                            train_model, prediction_mean, prediction_std = model.training(ds.X_train, ds.y_train,
                                                                                          ds.X_test,
                                                                                          ds.y_test, p)
                else:
                    train_model, prediction_mean, prediction_std = model.training(ds.X_train, ds.y_train,
                                                                                  ds.X_test,
                                                                                  ds.y_test, p)

                train_distribution = train_model(ds.X_train)
                train_mean = np.concatenate(train_distribution.mean().numpy(), axis=0)
                train_std = np.concatenate(train_distribution.stddev().numpy(), axis=0)

                save_results.save_uncertainty_csv(train_mean, train_std,
                                                  np.concatenate(ds.y_train, axis=0),
                                                  'avg' + res,
                                                  'train-' + model.name)

                mse = mean_squared_error(ds.y_test, prediction_mean)
                mae = mean_absolute_error(ds.y_test, prediction_mean)

                save_results.save_uncertainty_csv(prediction_mean, prediction_std,
                                                  np.concatenate(ds.y_test[:len(prediction_mean)], axis=0),
                                                  'avg' + res,
                                                  model.name)

                plot_training.plot_series_interval(np.arange(0, len(ds.y_test) - 1), ds.y_test, prediction_mean,
                                                   prediction_std,
                                                   label1="ground truth",
                                                   label2="prediction", title=model.name)

                plot_model(train_model, to_file='img/models/model_plot_' + model.name + '.png', show_shapes=True,
                           show_layer_names=True)
