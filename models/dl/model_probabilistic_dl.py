"""
Interface of a predictive probabilistic model with shared functionalities
Inherits from ModelInterface and ModelPorbabilistic class
"""
from models.model_probabilistic import ModelProbabilistic
from models.dl.model_interface_dl import ModelInterfaceDL

import numpy as np


class ModelProbabilisticDL(ModelProbabilistic, ModelInterfaceDL):
    def __init__(self, name):
        """
        Constructor of the Model Probabilistic class
        :param name: string: name of the model
        """
        super().__init__(name)
