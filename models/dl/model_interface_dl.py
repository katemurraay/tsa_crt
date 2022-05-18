from datetime import datetime
from models.model_interface import ModelInterface
import numpy as np

class ModelInterfaceDL(ModelInterface):
    def __init__(self, name):
        super().__init__(name)
        self.input_shape = None
        self.parameter_list = {}
        self.p = {}

        self.save_check = None
        self.es = None
        self.count_save = 0
        self.best_val_loss = np.Inf
    # Save the model into a file
    def save_model(self):
        if self.train_model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.train_model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf',
                              save_format="tf")
        self.count_save += 1


