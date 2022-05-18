from util.dataset import DatasetInterface

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf

import pickle


class ElectricityDataset(DatasetInterface):
    def __init__(self, name):
        super().__init__(name)
        pass

