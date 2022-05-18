from util import dataset, plot_training, save_results
import numpy as np
from models import LSTM
from keras.utils.vis_utils import plot_model
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
wins = [144]
hs = [2]
resources = ['cpu', 'mem']
clusters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
ITERATIONS = 10

train_splits = [0.8] 
tuning_rates = [6]
tuning = False

for tuning_rate in tuning_rates:
    for ts in train_splits:
        for win in wins:
            for res in resources:
                for h in hs:
                    for c in clusters:
                        mses, maes = [], []

                        if tuning:
                            experiment_name = 'tuned-LSTM-' + res + '-' + c + '-w' + str(win) + '-h' + str(h) + \
                                              '-tuning' + str(int(tuning_rate))
                        else:
                            experiment_name = 'tuned-LSTM-' + res + '-' + c + '-w' + str(win) + '-h' + str(h) + \
                                              '-ts' + str(int(ts * 100)) 

                        # Data creation and load
                        ds = dataset.Dataset(meta=False, filename='res_task_' + c + '.csv', input_window=win, horizon=h,
                                             resource=res, train_split=ts)
                        print(ds.name)
                        ds.dataset_creation()

                        ds.data_summary()

                        best_model, best_history, best_prediction_mean, best_prediction_std = None, None, None, None
                        best_mse = 100000

                        # Read the best hyperparameters
                        parameters = pd.read_csv("hyperparams/p_lstm-" + c + ".csv").iloc[0]

                        dense_act = 'relu'
                        if 'relu' in parameters['first_dense_activation']:
                            dense_act = 'relu'
                        elif 'tanh' in parameters['first_dense_activation']:
                            dense_act = 'tanh'
                            
                        p = {'first_conv_dim': parameters['first_conv_dim'],
                             'first_conv_activation': parameters['first_conv_activation'],
                             'first_conv_kernel': (parameters['first_conv_kernel'],),
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
                             }

                        print(p)

                        training_times, inference_times, tuning_times = [], [], []

                        for it in range(ITERATIONS):
                            print("RESOURCE:", res, "CLUSTER:", c, "ITERATION:", it, "HORIZON:", h, "WIN:", win)
                            model = LSTM.LSTMPredictor()
                            model.dataset = ds
                            model.name = experiment_name
                            start = datetime.now()

                            train_model, history, forecast, training_time, inference_time = \
                                model.training(p)
                            training_times.append(training_time)
                            inference_times.append(inference_time)

                            if tuning:
                                for i in range(int(ds.X_test.shape[0] / tuning_rate) - 1):
                                    train_model, history, tuning_time = model.tuning(
                                        ds.X_test[i * tuning_rate:(i + 1) * tuning_rate],
                                        ds.y_test[i * tuning_rate:(i + 1) * tuning_rate], p)
                                    tuning_times.append(tuning_time)

                            mse = mean_squared_error(ds.y_test, forecast)
                            mae = mean_absolute_error(ds.y_test, forecast)
                            mses.append(mse)
                            maes.append(mae)

                            save_results.save_output_csv(forecast, np.concatenate(ds.y_test[:len(forecast)], axis=0),
                                                         'avg' + res,
                                                         model.name + '-run-' + str(it))

                            if mse < best_mse:
                                best_mse = mse
                                best_prediction_mean = forecast
                                best_model = train_model
                                best_history = history

                        if tuning:
                            df_tuning_times = pd.DataFrame({'time': tuning_times})
                            df_tuning_times.to_csv("time/" + experiment_name + 'tuning_time.csv')
                        else:
                            df_training_time = pd.DataFrame({'time': training_times})
                            df_training_time.to_csv("time/" + experiment_name + 'training_time.csv')
                            df_inference_time = pd.DataFrame({'time': inference_times})
                            df_inference_time.to_csv("time/" + experiment_name + 'inference_time.csv')

                        forecast = best_prediction_mean
                        history = best_history
                        train_model = best_model

                        plot_training.plot_series(np.arange(0, len(ds.y_test) - 1), ds.y_test, forecast,
                                                  label1="ground truth",
                                                  label2="prediction", title=model.name)

                        plot_training.plot_loss(history, model.name)

                        plot_model(train_model, to_file='img/models/model_plot_' + model.name + '.png',
                                   show_shapes=True,
                                   show_layer_names=True)

                        plot_training.plot_series(np.arange(0, len(ds.y_test) - 1), ds.y_test, forecast,
                                                  label1="ground truth",
                                                  label2="prediction", title=model.name)

                        plot_training.plot_loss(history, model.name)

                        plot_model(train_model, to_file='img/models/model_plot_' + model.name + '.png',
                                   show_shapes=True,
                                   show_layer_names=True)

                        save_results.save_errors(mses, maes, model.name)
