"""
Class to customise PyTorch Callbacks
"""
from pytorch_lightning.callbacks import Callback
import numpy as np
import pytorch_lightning as pl

class CustomPytorchModelCheckpoint(Callback):       
    def __init__(self, model) -> None:
        """
        Constructor of the CustomSaveCheckpoint class
        :param model: model to monitor
        """
        super().__init__()
        self.dnn = model
        """ModelDLInterface: model to monitor"""
        self.monitor = 'val_loss'
        """string: metric to monitor"""
        self.counter = 0 
        """int: counter number of models saved"""

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """
        Check the validation loss at the end of the training epoch and save the model if the validation loss
        has improved
        :param trainer: pytorch_lightning.Trainer
        :param pl_module: pytorch_lightning.LightningModule
        :return: None
        """
        elogs = trainer.logged_metrics
        tensor_log = elogs[self.monitor]
        val_loss = tensor_log.item()
        
        if val_loss < self.dnn.best_val_loss and self.counter >=1 :
            print('\nNew best validation loss: ', val_loss)
            self.dnn.best_val_loss = val_loss
            self.dnn.model = self.dnn.temp_model
            self.dnn.save_model()
            print('Model save id: ', str(self.dnn.count_save - 1).zfill(4))
        self.counter += 1

      
       
        