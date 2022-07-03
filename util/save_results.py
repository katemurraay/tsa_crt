import pandas as pd
import numpy as np


def save_output_csv(preds, labels, feature, filename, bivariate=False):
    PATH = "res/output_" + filename + ".csv"
    if bivariate:
        labels = labels.reshape(-1, preds.shape[1])
        dct = {'avgcpu': preds[:, 0],
               'labelsavgcpu': labels[:, 0],
               'avgmem': preds[:, 1],
               'labelsavgmem': labels[:, 1]
               }
    else:
        try:
            preds = np.concatenate(list(preds), axis=0)
        except:
            pass
        try:
            labels = np.concatenate(list(labels), axis=0)
        except:
            pass
        # try:
        #     dct = {feature: np.concatenate(list(preds), axis=0),
        #            'labels': np.concatenate(list(labels), axis=0)}
        # except:
        dct = {feature: preds,
               'labels': labels}
        #     df = pd.DataFrame(dct)
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_uncertainty_csv(preds, std, labels, feature, filename, bivariate=False):
    PATH = "res/output_" + filename + ".csv"
    if bivariate:
        labels = labels.reshape(-1, preds.shape[1])
        dct = {'avgcpu': preds[:, 0],
               'stdavgcpu': std[:, 0],
               'labelsavgcpu': labels[:, 0],
               'avgmem': preds[:, 1],
               'stdavgmem': std[:, 1],
               'labelsavgmem': labels[:, 1],
               }
    else:
        try:
            dct = {feature: np.concatenate(list(preds), axis=0),
                   'std': np.concatenate(list(std), axis=0),
                   'labels': np.concatenate(list(labels), axis=0)}
        except:
            dct = {feature: preds,
                   'std': std,
                   'labels': np.concatenate(labels, axis=0)}

    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_params_csv(p, filename):
    PATH = "param/p_" + filename + ".csv"
    df = pd.DataFrame(p, index=[0])
    df.to_csv(PATH)


def save_bayes_csv(preds, min, max, labels, feature, filename):
    PATH = "res/vidp_" + filename + ".csv"
    dct = {feature: preds,
           'min': min,
           'max': max,
           'labels': labels}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_metrics_csv(mses, maes, rmse, mape, filename):
    PATH = "res/metrics_" + filename + ".csv"
    dct = {'MSE': mses,
           'MAE': maes, 
           'RMSE': rmse,
           'MAPE': mape}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)
