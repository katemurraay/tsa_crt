"""
Class to customise callbacks
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K


class CustomSaveCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, model):
        """
        Constructor of the CustomSaveCheckpoint class
        :param model: model to monitor
        """
        self.monitor = 'val_loss'
        """string: metric to monitor"""
        self.dnn = model
        """keras model: best model found"""


    def on_epoch_end(self, epoch, logs=None):
        """
        Check the validation loss at the end of the training epoch and save the model if the validation loss
        has improved
        :param epoch: int: epoch number
        :param logs: dict: dictionary of parameters and metrics to monitor
        :return: None
        """
        # print("LOGS", logs)
        print("BEST VALLOSS", self.dnn.best_val_loss, logs['val_loss'])
        if logs['val_loss'] < self.dnn.best_val_loss:
            print('New best validation loss at epoch ', epoch, ' :', logs['val_loss'])
            self.dnn.best_val_loss = logs['val_loss']
            self.dnn.model = self.dnn.temp_model
            self.dnn.save_model()
            print('Model save id ', str(self.dnn.count_save - 1).zfill(4))
