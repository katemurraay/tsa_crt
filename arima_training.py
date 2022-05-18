from util import dataset
from models import ARIMA
from util import plot_training
from util import save_results
import numpy as np
import pandas as pd


def main():    
    h = 2
    resources = ['cpu', 'mem']
    clusters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    tuned = False

    for resource in resources:
        for cluster in clusters:
            experiment_name = 'arima-' + cluster + '-' + resource + '-h' + str(h)

            # # Data creation and load
            ds = dataset.DatasetInterface(filename='res_task_' + cluster + '.csv', input_window=1, output_window=2,
                                          horizon=h, training_features=['avgcpu'],
                                          target_name=['avgcpu'], train_split_factor=0.8)

            ds.dataset_creation()
            ds.dataset_normalization(['minmax'])

            ds.data_summary()

            model = ARIMA.ARIMAPredictor()
            model.name = experiment_name

            if tuned:
                parameters = pd.read_csv("param/p_arima-"+cluster+"-h0.csv").iloc[0] 
                p = {'p': parameters['p'],
                     'd': parameters['d'],
                     'q': parameters['q'],
                     'P': parameters['P'],
                     'Q': parameters['Q'],
                     'D': parameters['D'],
                     'S': parameters['S'],
                     'selection': False,
                     'loop': parameters['loop'],
                     'horizon': parameters['horizon'],
                     'sliding_window': parameters['sliding_window']
                     }
            else:
                p = {'p': 2,
                     'd': 0,
                     'q': 2,
                     'P': 0,
                     'Q': 0,
                     'D': 0,
                     'S': 12,
                     'selection': False,
                     'loop': 1,
                     'horizon': 0,
                     'sliding_window': 288
                     }

            prediction_mean, prediction_std, train_model = model.training(ds.X_train, ds.y_train, ds.X_test, ds.y_test,
                                                                          p)

            if p['horizon'] > 0:
                prediction_mean = prediction_mean[:-p['horizon']]

            save_results.save_uncertainty_csv(prediction_mean, prediction_std,
                                              np.concatenate(ds.y_test[:len(prediction_mean)], axis=0),
                                              'avg'+resource,
                                              model.name)

            plot_training.plot_series_interval(np.arange(0, len(ds.y_test) - 1), ds.y_test, prediction_mean,
                                               prediction_std,
                                               label1="ground truth",
                                               label2="prediction", title=model.name)
            if not tuned:
                save_results.save_params_csv(model.parameter_list, model.name)


if __name__ == "__main__":
    main()
