import pandas as pd


def save_output_csv(preds, labels, feature, filename):
    PATH = "res/output_" + filename + ".csv"
    dct = {feature: preds,
           'labels': labels}
    print(dct)
    df = pd.DataFrame(dct)
    df.to_csv(PATH)


def save_uncertainty_csv(preds, std, labels, feature, filename):
    PATH = "res/output_" + filename + ".csv"
    dct = {feature: preds,
           'std': std,
           'labels': labels}
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


def save_errors(mses, maes, filename):
    PATH = "res/errors_" + filename + ".csv"
    dct = {'MSE': mses,
           'MAE': maes}
    df = pd.DataFrame(dct)
    df.to_csv(PATH)
