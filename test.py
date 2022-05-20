import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from models.dl.hbnn import HBNN
from models.stats.arima import ARIMA
from models.stats.garch import GARCH
from models.ml.svr import SVR
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
                experiment_name = 'svr-' + res + '-' + c + '-w' + str(win) + '-h' + str(h)

                # Data creation and load
                # ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                #                               horizon=h, training_features=['avgcpu', 'time', 'avgmem'],
                #                               target_name=['avg' + res, 'avgmem'], train_split_factor=0.8)

                ds = dataset.DatasetInterface(filename='res_task_' + c + '.csv', input_window=win, output_window=1,
                                              horizon=h, training_features=['avgcpu'],#, 'time', 'avgmem'],
                                              target_name=['avg' + res], train_split_factor=0.8)

                ds.dataset_creation()
                ds.dataset_normalization(['standard'])# , 'minmax', 'minmax'])

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
                # p = {'p': 2,
                #      'd': 0,
                #      'q': 2,
                #      'P': 0,
                #      'Q': 0,
                #      'D': 0,
                #      'S': 12,
                #      'loop': 0,
                #      'horizon': 2,
                #      }

                # # GARCH
                # p = {'p': 1,
                #      'q': 1,
                #      'loop': 1,
                #      'horizon': 0,
                #      'mean': 'LS',
                #      }
                #
                # # SVR
                # p = {'kernel': 'rbf',
                #      'degree': 3,
                #      'tol': 0.001,
                #      'C': 1.0,
                #      }

                print("RESOURCE:", res, "CLUSTER:", c, "HORIZON:", h, "WIN:", win)
                model = HBNN(experiment_name)
                # model = ARIMA(experiment_name)
                # model = GARCH(experiment_name)
                # model = SVR(experiment_name)

                model.ds = ds
                model.p = p
                model.create_model()

                # if len(files):
                #     for i in range(len(files)):
                #         path_weight = files[-(i + 1)][:-6]
                #         model.p['weight_file'] = path_weight
                #         try:
                #             model.load_model()
                #             prediction_mean, prediction_std = model.predict(ds.X_test)
                #         except:
                #             model.fit()
                # else:
                model.fit()
                print("Training complete")
                # model.hyperparametrization()
                # train_mean = model.evaluate()
                # preds = model.predict(ds.X_test)
                # labels = ds.y_test[h:len(preds) + h].reshape(-1, 1)
                #
                # if h > 0:
                #     preds = preds[:-h]
                #
                # try:
                #     print("MSE", mean_squared_error(labels, preds))
                #     print("MAE", mean_absolute_error(labels, preds))
                # except:
                #     labels = ds.y_test_array[h:len(preds) + h].reshape(-1, 1)
                #
                #     print("MSE", mean_squared_error(labels, preds))
                #     print("MAE", mean_absolute_error(labels, preds))
                #
                # plot_training.plot_series(np.arange(0, len(preds)), labels, preds, label1="ground truth",
                #                           label2="prediction", title=model.name)
                #
                # preds = np.array(preds).reshape(-1, 1)
                # preds = np.concatenate(preds, axis=0)
                # labels = np.concatenate(labels, axis=0)
                #
                # save_results.save_output_csv(preds, labels, 'avg' + res, model.name)
                #
                # save_results.save_params_csv(p, model.name)

                train_mean, train_std = model.evaluate()
                a = np.concatenate(ds.y_train[:len(train_mean)], axis=0).reshape(-1, 1)
                save_results.save_uncertainty_csv(train_mean, train_std,
                                                  a,
                                                  'avg' + res,
                                                  'train-' + model.name, bivariate=len(ds.target_name) > 1)

                prediction_mean, prediction_std = model.predict(ds.X_test)
                save_results.save_uncertainty_csv(prediction_mean, prediction_std,
                                                  np.concatenate(ds.y_test[:len(prediction_mean)], axis=0).reshape(-1,
                                                                                                                   1),
                                                  'avg' + res,
                                                  model.name, bivariate=len(ds.target_name) > 1)

                plot_training.plot_series_interval(np.arange(0, len(ds.y_test) - 1), ds.y_test, prediction_mean,
                                                   prediction_std,
                                                   label1="ground truth",
                                                   label2="prediction", title=model.name,
                                                   bivariate=len(ds.target_name) > 1)

                plot_model(model.model, to_file='img/models/model_plot_' + model.name + '.png', show_shapes=True,
                           show_layer_names=True)
