
from pytorch_lightning.callbacks import Callback
import numpy as np
import pytorch_lightning as pl

class CustomPytorchModelCheckpoint(Callback):       
    def __init__(self, model) -> None:
        super().__init__()
        self.dnn = model
        self.monitor = 'val_loss'
        self.counter = 0 

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        elogs = trainer.logged_metrics
        tensor_log = elogs['val_loss']
        val_loss = tensor_log.item()
        
        if val_loss < self.dnn.best_val_loss and self.counter >=1 :
            print('\nNew best validation loss: ', val_loss)
            self.dnn.best_val_loss = val_loss
            self.dnn.model = self.dnn.temp_model
            self.dnn.save_model()
            print('Model save id: ', str(self.dnn.count_save - 1).zfill(4))
        self.counter += 1

      
       
        