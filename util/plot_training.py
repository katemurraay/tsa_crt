import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("img/loss/loss_" + title + ".png")


def plot_series(series, series2, start=0, end=None, label1="", label2="", title="output"):
    plt.figure(figsize=(10, 6))
    plt.plot(series[start:end], label=label1)
    plt.plot(series2[start:end], 'o', label=label2)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/preds/" + title + ".png")


def plot_series_interval(series, series2, std, start=0, end=None, label1="", label2="", title="output"):
    plt.figure(figsize=(20, 10))
    min = np.array(series2) - 2 * np.array(std)
    min = np.squeeze(min)
    max = np.array(series2) + 2 * np.array(std)
    max = np.squeeze(max)
    plt.plot(series[start:end], label=label1)
    plt.plot(series2[start:end], 'o', label=label2)
    plt.fill_between(np.arange(len(series2)), min, max, alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Resource unit")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/preds/" + title + ".png")


def plot_bayes_series(series, series2, min, max, start=0, end=None, label1="", label2="", title="output"):
    plt.figure(figsize=(20, 10))
    plt.plot(series[start:end], label=label1)
    plt.plot(series2[start:end], 'o', label=label2)
    plt.fill_between(np.arange(len(series2)), min, max, alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("img/preds/" + title + ".png")
