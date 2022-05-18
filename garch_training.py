from util import dataset
from models import GARCH
from util import plot_training
from util import save_results
import numpy as np


def main():
    h = 2
    resources = ['cpu', 'mem']
    clusters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    for resource in resources:
        for cluster in clusters:
            experiment_name = 'garch-' + cluster + '-' + resource + '-h' + str(h)

            # Data creation and load

            ds = dataset.DatasetInterface(filename='res_task_' + cluster + '.csv', input_window=1, output_window=2,
                                          horizon=h, training_features=['avgcpu'],
                                          target_name=['avgcpu'], train_split_factor=0.8)

            ds.dataset_creation()
            ds.dataset_normalization(['minmax'])
            ds.data_summary()

            model = GARCH.GARCHPredictor()
            model.name = experiment_name

            p = {'p': 1,
                 'q': 1,
                 'selection': True,
                 'loop': 1,
                 'horizon': 0,
                 'mean': 'LS',
                 'sliding_window': 288
                 }

            prediction_mean, prediction_std, params, train_model = model.training(ds.X_train, ds.y_train,
                                                                                  ds.X_test, ds.y_test, p)

            if p['horizon'] > 0:
                prediction_mean = prediction_mean[:-p['horizon']]

            plot_training.plot_series_interval(np.arange(0, len(ds.y_test) - 1), ds.y_test, prediction_mean,
                                               prediction_std, label1="ground truth",
                                               label2="prediction", title=model.name)

            save_results.save_uncertainty_csv(prediction_mean, prediction_std,
                                              np.concatenate(ds.y_test[:len(prediction_mean)], axis=0),
                                              'avg' + resource, model.name)

            p = params
            save_results.save_params_csv(p, model.name)


if __name__ == "__main__":
    main()
