from tabnanny import verbose
import eagerpy as ep
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import optuna
from optuna.trial import TrialState
import eagerpy as ep
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from darts.models import TFTModel
from darts.metrics import mse
from models.model_interface import ModelInterface
from util import custom_keras
from torchmetrics import MeanSquaredError
from darts.metrics import mse

class TFT(ModelInterface):
    def __init__(self, name):
        super().__init__(name)
        self.p ={ 'epochs': 500, 
                'input_chunk_length': 1,
                'hidden_layer_dim': 64,
                'num_lstm_layers' : 4,
                'num_attention_heads': 2,
                'dropout_rate': 0.0,
                'batch_size': 64,
                'output_chunk_length': 1,
                'patience': 50,
        }
        self.parameter_list=  {  "input_chunk_length":[1, 32, 64], 
                "hidden_size":[32, 64, 128], 
                "lstm_layers":[2, 4], 
                "num_attention_heads":[2, 3, 4], 
                "dropout":[0.0, 0.05, 0.1], 
                "batch_size":[32, 64], 
                'output_chunk_length': [1 ,32, 64],
                "n_epochs":[1000]
                }

       
        self.RAND = 42           
        self.N_JOBS = 3
        use_cuda = torch.cuda.is_available()
        self.use_device = "cuda:0" if use_cuda else "cpu"


    def create_model(self):
        """TODO: add pytorch model checkpointing resembling custom_keras.CustomSaveCheckpoint(self) """
        my_stopper = EarlyStopping(
                                    monitor="train_loss",
                                    patience=self.p['patience'],
                                    min_delta=0.00,
                                    verbose= 0,
                                    mode='min',
                                )
        self.temp_model = TFTModel(input_chunk_length=self.p['input_chunk_length'],
                    output_chunk_length= self.p['output_chunk_length'],
                    hidden_size=self.p['hidden_layer_dim'],
                    lstm_layers= self.p['num_lstm_layers'],
                    num_attention_heads= self.p['num_attention_heads'],
                    dropout= self.p['dropout_rate'],
                    batch_size= self.p['batch_size'],
                    n_epochs= self.p['epochs'],
                    likelihood=None, 
                    loss_fn= torch.nn.MSELoss(),
                    torch_metrics = MeanSquaredError(),
                    random_state= self.RAND, 
                    pl_trainer_kwargs={"callbacks": [my_stopper]},
                    torch_device_str = self.use_device,
                    force_reset=True,
                    
                    )
        
    def fit(self):
        self.temp_model.fit(self.ds.ts_train, future_covariates=self.ds.tcov,  val_series =self.ds.ts_test, val_future_covariates=self.ds.tcov, verbose= True)   
    def fit_predict(self, X):
        self.temp_model.fit(self.ds.ts_train, future_covariates=self.ds.tcov,  val_series =self.ds.ts_test, val_future_covariates=self.ds.tcov, verbose = True) 
        predictions = self.temp_model.predict(n=len(X))
        return predictions
        
    def predict(self, X):
        if self.temp_model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        else: 
            predictions = self.temp_model.predict(n=len(X))
            return predictions


    def __tft_darts_grid_search(self):
        my_stopper = EarlyStopping(
                                    monitor="val_loss",
                                    patience=self.p['patience'],
                                    min_delta=0.00,
                                    verbose= 0,
                                    mode='min',
                                )
        model = TFTModel(input_chunk_length=self.p['input_chunk_length'],
                    output_chunk_length= self.p['output_chunk_length'],
                    hidden_size=self.p['hidden_layer_dim'],
                    lstm_layers= self.p['num_lstm_layers'],
                    num_attention_heads= self.p['num_attention_heads'],
                    dropout= self.p['dropout_rate'],
                    batch_size= self.p['batch_size'],
                    n_epochs= self.p['epochs'],
                    likelihood=None, 
                    loss_fn= torch.nn.MSELoss(),
                    torch_metrics = MeanSquaredError(),
                    random_state= self.RAND, 
                    pl_trainer_kwargs={"callbacks": [my_stopper]},
                    torch_device_str =self.use_device,
                    force_reset=True,
                    
                    )
        res = model.gridsearch(    
                            parameters=self.parameter_list,
                            series=self.ds.ts_train, 
                            future_covariates=self.ds.tcov, 
                            val_series=self.ds.ts_test,   
                            start=0.1,               
                            last_points_only=False, 
                            metric=mse, 
                            reduction=np.mean, 
                            n_jobs=self.N_JOBS, 
                            n_random_samples=0.99,     
                            verbose=True)
        self.temp_model, dict_bestparams = res
        print(dict_bestparams)
    
    def __optuna_objective(self, trial):
        input_chunk = trial.suggest_categorical('input_chunk_length', self.parameter_list['input_chunk_length'])
        output_chunk =trial.suggest_categorical('output_chunk_length', self.parameter_list['output_chunk_length'])
        hidden_size = trial.suggest_categorical('hidden_size', self.parameter_list['hidden_size'])
        attention_heads = trial.suggest_categorical('num_attentions_heads',self.parameter_list['num_attention_heads'])
        lstm_layers = trial.suggest_categorical('lstm_layers', self.parameter_list['lstm_layers'])
        dropout = trial.suggest_categorical('dropout_rate', self.parameter_list['dropout'])
        batch_size = trial.suggest_categorical('batch_size', self.parameter_list['batch_size'])
        epochs = trial.suggest_categorical('epochs', self.parameter_list['n_epochs'])
        
        print('Trial Params: {}'.format(trial.params))
        my_stopper = EarlyStopping(
                                    monitor="val_loss",
                                    patience=self.p['patience'],
                                    min_delta=0.00,
                                    verbose= 0,
                                    mode='min',
                                )
        self.temp_model = TFTModel(input_chunk_length=input_chunk,
                    output_chunk_length=output_chunk,
                    hidden_size= hidden_size,
                    lstm_layers= lstm_layers,
                    num_attention_heads= attention_heads,
                    dropout= dropout,
                    batch_size= batch_size,
                    n_epochs= epochs,
                    likelihood=None, 
                    loss_fn= torch.nn.MSELoss(),
                    torch_metrics = MeanSquaredError(),
                    random_state= self.RAND, 
                    pl_trainer_kwargs={"callbacks": [my_stopper]},
                    torch_device_str = self.use_device,
                    force_reset=True,
                    
                    )
        preds = self.fit_predict(self.ds.ts_test)
        return mse(self.ds.ts_test, preds)
 
    def hyperparametrization(self):
        study = optuna.create_study(study_name="TFT_Optimization", direction="minimize")
        study.optimize(self.__optuna_objective, n_trials=100, show_progress_bar=True)
        print('\nBEST PARAMS: \n{}'.format(study.best_params))
        print('\nBEST VALUE:\n{}'.format(study.best_value))
        df = study.trials_dataframe()
        filename = 'talos/' + self.name +'.csv'
        df.to_csv(filename)
    
       

    
   
   