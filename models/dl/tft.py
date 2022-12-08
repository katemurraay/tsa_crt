"""
TFT-based model
Inherits from ModelInterfaceDL
"""

import numpy as np
import pandas as pd
import optuna
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.models import TFTModel
from models.dl.model_interface_dl import ModelInterfaceDL
from torchmetrics import MeanSquaredError
from darts.metrics import mse
from darts.timeseries import concatenate
from util import custom_pytorch

class TFT(ModelInterfaceDL):
    def __init__(self, name):
        """
        Constructor of the Model Interface class
        :param name: string: name of the model
        """
        super().__init__(name)
        self.parameter_list=  {
                "input_chunk_length":[30], 
                "hidden_layer_dim":[16, 32, 64], 
                "num_lstm_layers":[2, 3, 4, 5], 
                "num_attention_heads":[2, 4, 5, 7], 
                "dropout_rate'":[0.0, 0.05, 0.1], 
                "batch_size":[256], 
                'output_chunk_length': [1],
                "epochs":[2000],
                 "lr": [1e-3, 1e-4, 1e-5],
                 "patience":[50],
                'optimizer': ['adam', 'nadam', 'rmsprop'],
                'feed_forward': ['GatedResidualNetwork', 'GLU', 'Bilinear', 'ReLU'],
                }
        """dict parameter_list: Dictionary of hyperparameter configuration of the model"""
        self.p ={
                'input_chunk_length': 30,
                'hidden_layer_dim': 128,
                'num_lstm_layers' : 5,
                'num_attention_heads': 5,
                'dropout_rate': 0.05,
                'batch_size': 256,
                'output_chunk_length': 1,
                'epochs': 200, 
                'lr': 1e-3,
                'patience': 50,
                'optimizer': 'adam',
                'feed_forward': 'GatedResidualNetwork',
                }
        """dict p: Dictionary of hyperparameters search space"""
        
        self.pl_trainer_kwargs = self.__set_pl_trainer_kwargs()
        """dict pl_trainer_kwargs: Dictionary of PyTorch Lightning Trainer keyword arguments"""
        self.RAND = 42     
        """int RAND: Seeds the random initialisation of weights"""

    def __set_pl_trainer_kwargs(self):
        """
        Sets the PyTorch Lightning Trainer Keyword Arguments
        :return dict pl_trainer_kwargs: Dictionary of the keyword arguments for the PyTorch Lightning Trainer
        """
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
        """
        Create an instance of the model. This function contains the definition and the library of the model
        :return: None
        """
        self.temp_model = TFTModel(
                        input_chunk_length=self.p['input_chunk_length'],
                        output_chunk_length= self.p['output_chunk_length'],
                        hidden_size=self.p['hidden_layer_dim'],
                        lstm_layers= self.p['num_lstm_layers'],
                        num_attention_heads= self.p['num_attention_heads'],
                        dropout = self.p['dropout_rate'],
                        batch_size = self.p['batch_size'],
                        n_epochs= self.p['epochs'],
                        likelihood=None, 
                        loss_fn= torch.nn.MSELoss(),
                        full_attention=False,
                        torch_metrics = MeanSquaredError(),
                        random_state= self.RAND, 
                        force_reset= True,
                        feed_forward=self.p['feed_forward'],
                        pl_trainer_kwargs = self.pl_trainer_kwargs,
                        add_relative_index= False,
                    )
        if self.p['optimizer'] =='rmsprop':
            opt = torch.optim.RMSprop
        elif self.p['optimizer'] =='nadam':
            opt = torch.optim.NAdam
        else:
            opt = torch.optim.Adam
        
        self.temp_model.optimizer_cls = opt
        self.temp_model.optimizer_kwargs={"lr": self.p['lr']}
        self.temp_model.lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
        
    def fit(self, use_covariates = True):
        """
        Training of the model on darts.TimeSeries Training Data
        :return: None
        """
        my_stopper = EarlyStopping(
                                    monitor="val_loss",
                                    patience=self.p['patience'],
                                    verbose= 0,
                                    mode='min',
                                )
        checkpoint = custom_pytorch.CustomPytorchModelCheckpoint(self)
        self.pl_trainer_kwargs["callbacks"] = [my_stopper, checkpoint]
        self.temp_model.trainer_params =self.pl_trainer_kwargs
        if use_covariates:
            self.temp_model.fit(self.ds.ts_train,  future_covariates=self.ds.f_cov,  val_series =self.ds.ts_val, val_future_covariates=self.ds.f_cov, verbose= 1)   
        else:
            print('Using Relative Index')
            self.temp_model.add_relative_index = True
            self.temp_model.fit(self.ds.ts_train, val_series =self.ds.ts_val, verbose=1)
        self.model = checkpoint.dnn.model
        return self.model
        
   
    def fit_predict(self, X):
        """
        Training of the model and Inference step on the samples X
        :param: darts.TimeSeries X: Values to be Predicted 
        :return: darts.TimeSeries predictions: Predictons of the Model
        """
        my_stopper = EarlyStopping(
                                    monitor="val_loss",
                                    patience=self.p['patience'],
                                    verbose= 0,
                                    mode='min',
                                )
        checkpoint = custom_pytorch.CustomPytorchModelCheckpoint(self)
        self.pl_trainer_kwargs["callbacks"] = [my_stopper, checkpoint]
        self.temp_model.trainer_params =self.pl_trainer_kwargs
        self.temp_model.fit(self.ds.ts_train,  future_covariates=self.ds.f_cov,  val_series =self.ds.ts_val,  val_future_covariates=self.ds.f_cov, verbose= 1)   
        self.model = checkpoint.dnn.model
        predictions = self.temp_model.predict(n=len(X))
        return predictions
        
    def predict(self, X):
        """
        Inference step on the samples X
        :param: darts.TimeSeries X: Values to be Predicted 
        :return: darts.TimeSeries predictions: Predictons of the Model
        """
        if self.model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        else: 
            predictions = self.model.predict(n=len(X))
            return predictions
  
    def __optuna_objective(self, trial):
        """
        Custom Function of Optuna which represents a single execution of a trial
        :param dict trial: Represents the hyperparameters being examined
        :return float mse: MeanSquaredError of the Model's Predictive Performance on the Validation Set
        """
        input_chunk = trial.suggest_categorical('input_chunk_length', self.parameter_list['input_chunk_length'])
        output_chunk =trial.suggest_categorical('output_chunk_length', self.parameter_list['output_chunk_length'])
        hidden_size = trial.suggest_categorical('hidden_layer_dim', self.parameter_list['hidden_layer_dim'])
        lstm_layers = trial.suggest_categorical('num_lstm_layers', self.parameter_list['num_lstm_layers'])
        attention_heads = trial.suggest_categorical('num_attention_heads',self.parameter_list['num_attention_heads'])
        dropout = trial.suggest_categorical('dropout_rate', self.parameter_list['dropout_rate'])
        batch_size = trial.suggest_categorical('batch_size', self.parameter_list['batch_size'])
        epochs = trial.suggest_categorical('epochs', self.parameter_list['epochs'])
        learning_rate = trial.suggest_categorical('lr', self.parameter_list['lr'])
        optimizer = trial.suggest_categorical('optimizer', self.parameter_list['optimizer'])
        feed_forward  = trial.suggest_categorical('feed_forward', self.parameter_list['feed_forward'])
        print('Trial Params: {}'.format(trial.params))
        self.pl_trainer_kwargs = self.__set_pl_trainer_kwargs()
        self.temp_model = TFTModel(
                    input_chunk_length=input_chunk,
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
                    force_reset=True,
                    pl_trainer_kwargs = self.pl_trainer_kwargs,
                    feed_forward= feed_forward
                    )
        if optimizer =='rmsprop':
            opt = torch.optim.RMSprop
        elif optimizer =='nadam':
            opt = torch.optim.NAdam
        else:
            opt = torch.optim.Adam
        
        self.temp_model.optimizer_cls = opt
        self.temp_model.optimizer_kwargs={"lr": learning_rate}
        self.temp_model.lr_scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
        preds = self.fit_predict(self.ds.ts_val)
        return mse(self.ds.ts_val, preds)
 
    def hyperparametrization(self):
        """
        Search the best parameter configuration using Optuna
        :return: None
        """
        study = optuna.create_study(study_name="TFT_Optimization", direction="minimize", sampler= optuna.samplers.GridSampler(self.parameter_list))
        study.optimize(self.__optuna_objective, n_trials=100, show_progress_bar=True)
        print('\nBEST PARAMS: \n{}'.format(study.best_params))
        print('\nBEST VALUE:\n{}'.format(study.best_value))
        df = study.trials_dataframe()
        filename = 'talos/' + self.name +'.csv'
        df.to_csv(filename)
    
    def save_model(self):
        """
        Save the model into a file in self.saved_models directory
        :return: boolean: 1 if saving operating is successful, 0 otherwise
        """
        if self.model is None:
            print("ERROR: the model must be available before saving it")
            return
        path = (self.model_path + self.name + str(self.count_save).zfill(4) + '_model.pth.tar')
        self.model.save_model(path)
        self.count_save += 1
     
    def load_model(self):
        """
        Load the model from a file
        :return: boolean: 1 if loading operating is successful, 0 otherwise
        """
        count_save = self.count_save - 1
        path = (self.model_path + self.name + str(count_save).zfill(4) + '_model.pth.tar')
        if self.model is None:
            print("ERROR: the model must be available before loading it")
            return
        self.temp_model = self.model.load_model(path)
    def tune(self, X):
        """
        Tune the models with new available samples (X, y)
        :param X: nparray: Training features
        :param y: nparray: Training labels
        :return: None
        """
        my_stopper = EarlyStopping(
                                    monitor="train_loss",
                                    patience=self.p['patience'],
                                    verbose= 0,
                                    mode='min',
                                )
        checkpoint = custom_pytorch.CustomPytorchModelCheckpoint(self)
        self.pl_trainer_kwargs["callbacks"] = [my_stopper, checkpoint]
        self.temp_model.trainer_params =self.pl_trainer_kwargs
        self.temp_model.fit(X,  verbose= 1)   
        self.model = checkpoint.dnn.model    

    def __tft_backtest(self):
        if self.temp_model is None:
            print("ERROR: the model needs to be trained before predict")
            return
        else: 
            h_forecasts = self.temp_model.historical_forecasts(series=self.ds.ts_ttrain,  
                                                        past_covariates=None, 
                                                        future_covariates=None, 
                                                        num_samples=1, 
                                                        train_length=None, 
                                                        start= 0.02, 
                                                        forecast_horizon=1, 
                                                        stride=1, 
                                                        retrain=False, 
                                                        overlap_end=False, 
                                                        last_points_only=False, 
                                                        verbose=True)
            h_forecasts = concatenate(h_forecasts)
            return h_forecasts
    def evaluate(self):
        """
        Evaluate the model on the training set ds.ts_train
        :return: np.array: predictions: predictions of the trained model on the ds.ts_train set
        """
        return self.predict(self.ds.ts_ttrain)
        

    def evaluate_backtest(self):
        return self.__tft_backtest()
    
    def training(self, p, X_test):
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 1
        timings=np.zeros((repetitions,1))
        
        for _ in range(10):
            self.p = p
            self.create_model()
            
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                train_model = self.fit(use_covariates = False)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                timings[rep] = curr_time

        self.train_time = np.sum(timings) / repetitions
        for _ in range(10):
            to_predict = X_test
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                predictions = self.predict(to_predict)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_time = np.sum(timings) / repetitions
        self.inference_time = mean_time / len(to_predict)
        
        return  predictions, train_model
   
   