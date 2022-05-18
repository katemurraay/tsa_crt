"""
Interface of a predictive DL model with shared functionalities
Inherits from ModelInterface class
"""

from models.model_interface import ModelInterface
import numpy as np


class ModelInterfaceDL(ModelInterface):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)

        self.count_save = 0
        """int: counter of the model saving"""
        self.best_val_loss = np.Inf
        """float: best validation found so far"""

    def save_model(self):
        """
        Save the model into a file in self.saved_models directory
        :return: boolean: 1 if saving operating is successful, 0 otherwise
        """
        if self.temp_model is None:
            print("ERROR: the model must be available before saving it")
            return
        self.temp_model.save(self.model_path + self.name + str(self.count_save).zfill(4) + '_model.tf',
                             save_format="tf")
        self.count_save += 1
        return 1
