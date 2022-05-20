import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.__history['loss'])
    plt.plot(history.__history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("img/loss/loss_" + title + ".png")


def plot_series(time, series, series2, format="-", start=0, end=None, label1="", label2="", title="output",
                bivariate=False):
    plt.figure(figsize=(10, 6))
    if bivariate:
        plt.plot(series[start:end, 0], label=label1+'CPU')
        plt.plot(series2[start:end, 0], 'o', label=label2+'CPU')
        plt.plot(series[start:end, 1], label=label1+'MEM')
        plt.plot(series2[start:end, 1], 'o', label=label2+'MEM')
    else:

        plt.plot(series[start:end], label=label1)
        plt.plot(series2[start:end], 'o', label=label2)
    plt.xlabel("Time")
    plt.ylabel("Resource unit")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/preds/" + title + ".png")


def plot_series_interval(time, series, series2, std, format="-", start=0, end=None, label1="", label2="",
                         title="output", bivariate=False):
    plt.figure(figsize=(20, 10))
    if bivariate:
        min_array = np.array(series2) - 2 * np.array(std)
        min_array = np.squeeze(min_array)
        max_array = np.array(series2) + 2 * np.array(std)
        max_array = np.squeeze(max_array)
        plt.plot(series[start:end, :, 0], label=label1 + 'CPU')
        plt.plot(series2[start:end, 0], 'o', label=label2 + 'CPU')
        plt.plot(series[start:end, :, 1], label=label1 + 'MEM')
        plt.plot(series2[start:end, 1], 'o', label=label2 + 'MEM')
        plt.fill_between(np.arange(len(series2)), min_array[start:end, 0], max_array[start:end, 0], alpha=0.2)
        plt.fill_between(np.arange(len(series2)), min_array[start:end, 1], max_array[start:end, 1], alpha=0.2)

    else:
        series = np.array(series).reshape(-1, 1)
        series2 = np.array(series2).reshape(-1, 1)
        std = np.array(std).reshape(-1, 1)
        min_array = np.array(series2) - 2 * np.array(std)
        min_array = np.squeeze(min_array)
        max_array = np.array(series2) + 2 * np.array(std)
        max_array = np.squeeze(max_array)
        plt.plot(series[start:end], label=label1)
        plt.plot(series2[start:end], 'o', label=label2)

        plt.fill_between(np.arange(series2.shape[0]), min_array, max_array, alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Resource unit")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/preds/" + title + ".png")


def plot_bayes_series(time, series, series2, min, max, format="-", start=0, end=None, label1="", label2="",
                      title="output"):
    plt.figure(figsize=(20, 10))
    plt.plot(series[start:end], label=label1)
    plt.plot(series2[start:end], 'o', label=label2)
    plt.fill_between(np.arange(len(series2)), min, max, alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/preds/" + title + ".png")
