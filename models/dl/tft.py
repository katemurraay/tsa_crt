import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from darts.models import TFTModel
from darts.metrics import mse
from models.dl.model_interface_dl import ModelInterfaceDL
from torchmetrics import MeanSquaredError
from darts.metrics import mse
from util import custom_pytorch
import warnings
warnings.filterwarnings("ignore")

class TFT(ModelInterfaceDL):
    def __init__(self, name):
        super().__init__(name)
        self.p ={ 'epochs': 500, 
                'input_chunk_length': 30,
                'hidden_layer_dim': 128,
                'num_lstm_layers' : 4,
                'num_attention_heads': 5,
                'dropout_rate': 0.05,
                'batch_size': 128,
                'output_chunk_length': 1,
                'patience': 50,
                'lr': 1e-4,
                'optimizer': 'adam',
        }
        self.parameter_list=  {
                "input_chunk_length":[30], 
                "hidden_size":[32, 64, 128], 
                "lstm_layers":[2, 3, 4, 5], 
                "num_attention_heads":[2, 3, 5, 7], 
                "dropout":[0.0, 0.05, 0.1], 
                "batch_size":[256], 
                'output_chunk_length': [1],
                "n_epochs":[1000],
                 "lr": [1e-3, 1e-4, 1e-5],
                'optimizer': ['adam', 'nadam', 'rmsprop'],
                }
        self.pl_trainer_kwargs = self.__set_pl_trainer_kwargs()
        self.RAND = 42           

        

    def __set_pl_trainer_kwargs(self):
       
        pl_trainer_kwargs = {"enable_model_summary":False, "enable_checkpointing": False, "logger": False, "weights_summary" : None}
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            pl_trainer_kwargs["accelerator"]= "gpu"
            pl_trainer_kwargs["gpus"] = -1
            pl_trainer_kwargs["auto_select_gpus"] = True
        else:
            pl_trainer_kwargs["accelerator"]= "cpu"
        return pl_trainer_kwargs
    def create_model(self):
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
                    optimizer_kwargs={"lr": self.p['lr']}, 
                    torch_metrics = MeanSquaredError(squared=True),
                    random_state= self.RAND, 
                    force_reset=True,
                    pl_trainer_kwargs = self.pl_trainer_kwargs,
                    lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau,
                    )
        if self.p['optimizer'] =='rmsprop':
            opt = torch.optim.RMSprop
        elif self.p['optimizer'] =='nadam':
            opt = torch.optim.NAdam
        else:
            opt = torch.optim.Adam
        
        self.temp_model.optimizer_cls = opt
        
    def fit(self):
        my_stopper = EarlyStopping(
                                    monitor="val_loss",
                                    patience=self.p['patience'],
                                    verbose= 0,
                                    mode='min',
                                )
        checkpoint = custom_pytorch.CustomPytorchModelCheckpoint(self)
        self.pl_trainer_kwargs["callbacks"] = [my_stopper, checkpoint]
        self.temp_model.trainer_params =self.pl_trainer_kwargs
        #self.temp_model.fit(self.ds.ts_train, past_covariates = self.ds.p_cov, future_covariates=self.ds.f_cov,  val_series =self.ds.ts_val, val_past_covariates=self.ds.p_cov, val_future_covariates=self.ds.f_cov, verbose= 1)   
        self.temp_model.fit(self.ds.ts_train,  future_covariates=self.ds.f_cov,  val_series =self.ds.ts_val,  val_future_covariates=self.ds.f_cov, verbose= 1)   
        
        self.model = checkpoint.dnn.model
    def fit_predict(self, X):
        my_stopper = EarlyStopping(
                                    monitor="val_loss",
                                    patience=self.p['patience'],
                                    min_delta=0.00,
                                    verbose= 0,
                                    mode='min',
                                )
        checkpoint = custom_pytorch.CustomPytorchModelCheckpoint(self)
        self.pl_trainer_kwargs["callbacks"] = [my_stopper, checkpoint]
        self.temp_model.trainer_params =self.pl_trainer_kwargs
        self.temp_model.fit(self.ds.ts_train, past_covariates = self.ds.p_cov, future_covariates=self.ds.f_cov,  val_series =self.ds.ts_val, val_past_covariates=self.ds.p_cov, val_future_covariates=self.ds.f_cov, verbose= 1)   
        #self.temp_model.fit(self.ds.ts_train, future_covariates=self.ds.f_cov,  val_series =self.ds.ts_val, val_future_covariates=self.ds.f_cov, verbose= 1)   
        predictions = self.temp_model.predict(n=len(X))
        return predictions
        
    def predict(self, X):
        if self.temp_model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        else: 
            predictions = self.temp_model.predict(n=len(X))
            return predictions


   
    def __optuna_objective(self, trial):
        input_chunk = trial.suggest_categorical('input_chunk_length', self.parameter_list['input_chunk_length'])
        output_chunk =trial.suggest_categorical('output_chunk_length', self.parameter_list['output_chunk_length'])
        hidden_size = trial.suggest_categorical('hidden_size', self.parameter_list['hidden_size'])
        attention_heads = trial.suggest_categorical('num_attentions_heads',self.parameter_list['num_attention_heads'])
        lstm_layers = trial.suggest_categorical('lstm_layers', self.parameter_list['lstm_layers'])
        dropout = trial.suggest_categorical('dropout_rate', self.parameter_list['dropout'])
        batch_size = trial.suggest_categorical('batch_size', self.parameter_list['batch_size'])
        epochs = trial.suggest_categorical('epochs', self.parameter_list['n_epochs'])
        learning_rate = trial.suggest_categorical('learning_rate', self.parameter_list['lr'])
        optimizer = trial.suggest_categorical('optimizer', self.parameter_list['optimizer'])
        print('Trial Params: {}'.format(trial.params))
        self.pl_trainer_kwargs = self.__set_pl_trainer_kwargs()
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
                    optimizer_kwargs={"lr": learning_rate}, 
                    torch_metrics = MeanSquaredError(),
                    random_state= self.RAND, 
                    force_reset=True,
                    pl_trainer_kwargs = self.pl_trainer_kwargs,
                    )
        if optimizer =='rmsprop':
            opt = torch.optim.RMSprop
        elif optimizer =='nadam':
            opt = torch.optim.NAdam
        else:
            opt = torch.optim.Adam
        
        self.temp_model.optimizer_cls = opt
        preds = self.fit_predict(self.ds.ts_val)
        return mse(self.ds.ts_val, preds)
 
    def hyperparametrization(self):
        study = optuna.create_study(study_name="TFT_Optimization", direction="minimize", sampler= optuna.samplers.TPESampler())
        study.optimize(self.__optuna_objective, n_trials=100, show_progress_bar=True)
        print('\nBEST PARAMS: \n{}'.format(study.best_params))
        print('\nBEST VALUE:\n{}'.format(study.best_value))
        df = study.trials_dataframe()
        filename = 'talos/' + self.name +'.csv'
        df.to_csv(filename)
    
    def save_model(self):
        if self.model is None:
            print("ERROR: the model must be available before saving it")
            return
        path = (self.model_path + self.name + str(self.count_save).zfill(4) + '_model.pth.tar')
        self.model.save_model(path)
        self.count_save += 1
     
    def load_model(self):
        path = (self.model_path + self.name + str(self.count_save).zfill(4) + '_model.pth.tar')
        self.temp_model = self.model.load_model(path)
       

    
   
   