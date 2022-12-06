from models.ml.rf import RF
from models.ml.svr import SVR
from models.ml.knn import KNN
from models.dl.tcn import TCN
from models.dl.tft import TFT
from models.dl.lstm_gru_hybrid import LSTM_GRU
from models.dl.gru import GRU
from models.dl.lstm import LSTM
from util import dataset_binance
import sys
def hyperparameter_test(model_name, cryptos, targets):
    scaling = ['minmax']
    for t in targets:
        for c in cryptos:
            experiment_name = model_name +'-' + c + '-' + t
            ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=30, output_window=1,
                                                        horizon=0, training_features=['close'],
                                                        target_name=['close'], train_split_factor=0.8)
            df, diff_df = ds.differenced_dataset()
            ds.df = diff_df
            if model_name == 'RF':
                model = RF(experiment_name)
                model.parameter_list ={'n_estimators': [200, 400, 500, 800, 1000],
                                        'criterion': ["squared_error"],
                                        'max_depth': [10, 50, 70, 100],
                                        'min_samples_leaf': [1, 2, 4],
                                        'min_samples_split': [2, 5, 10],
                                        'max_features': ["auto", "sqrt"],
                                        'bootstrap': [True, False]
                                        }
            elif model_name =='SVR':
                model = SVR(experiment_name)
                model.parameter_list = {'kernel': ('linear', 'poly', 'rbf', 'sigmoid'),
                               'degree': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                               'gamma' : ['auto', 0.1, 1, 0.0, 0.001],
                               'C': [0.1,1, 10, 100],

                               }
            elif model_name =='TCN':
                model = TCN(experiment_name)
                model.parameter_list = {
                                        'conv_filter': [16, 32, 64],
                                        'conv_kernel': [4, 8, 16],
                                        'conv_activation': ['relu'],
                                        'dilation_rate': [1, 4, 8],
                                        'dropout_rate': [0.0 , 0.05 , 0.1],
                                        'dense_dim': [32, 64],
                                        'dense_activation': ['relu'],
                                        'batch_size': [256],
                                        'epochs': [1000],
                                        'patience': [50],
                                        'optimizer': ['adam'],
                                        'lr': [1E-3, 1E-4, 1E-5],
                                    } 
            elif model_name == 'TFT':
                model = TFT(experiment_name)
                model.parameter_list = {
                                        "input_chunk_length":[30], 
                                        "hidden_size":[32, 64, 128], 
                                        "lstm_layers":[2, 3, 4, 5], 
                                        "num_attention_heads":[2, 3, 5, 7], 
                                        "dropout":[0.0, 0.05, 0.1], 
                                        "batch_size":[256], 
                                        'output_chunk_length': [1],
                                        "n_epochs":[1000],
                                        "lr": [1e-3, 1e-4, 1e-5],
                                        'optimizer': ['adam'],          
                                        }
            elif model_name =='HYBRID':
                        model =LSTM_GRU(experiment_name)
                        model.parameter_list ={

                                                'lstm_dim_1': [50, 75],
                                                'lstm_activation': ['relu'],
                                                'dropout_rate_1': [0.05, 0.1],
                                                'lstm_dim_2': [50, 75],
                                                'dense_dim_1': [32, 64],
                                                'dense_activation': ['relu'],
                                                'gru_dim':[50,75],
                                                'gru_activation': ['relu'],
                                                'dropout_rate_2': [0.0, 0.05, 0.1],
                                                'dense_dim_2': [ 32, 64],
                                                'epochs': [200],
                                                'patience': [20],
                                                'batch_size': [256],
                                                'optimizer': ['adam'],
                                                'lr': [1E-3, 1E-4, 1E-5],


                        }
            elif model_name =='LSTM':
                        model =LSTM(experiment_name)
                        model.parameter_list ={
                                'first_conv_dim': [32, 64],
                               'first_conv_kernel': [3, 5, 7],
                               'first_conv_activation': ['relu'],
                               'first_lstm_dim': [50, 75],
                               'first_dense_dim': [16, 32, 64],
                               'first_dense_activation': ['relu', 'tanh'],
                               'batch_size': [256],
                               'epochs': [200],
                               'patience': [20],
                               'optimizer': ['adam'],
                               'lr': [1E-3, 1E-4, 1E-5],
             }
            elif model_name =='GRU':
                        model = GRU(experiment_name)
                        model.parameter_list = {
                                'first_gru_dim': [50, 75, 100],
                               'gru_activation': ['relu'],
                               'first_dense_dim': [16, 32, 64, 100],
                               'first_dense_activation': ['relu'],
                               'batch_size': [256],
                               'epochs': [200],
                               'patience': [20],
                               'optimizer': ['adam', 'nadam','rmsprop'],
                               'lr': [1E-3, 1E-4, 1E-5],
                               }
            else:
                model = KNN(experiment_name)
                model.parameter_list = {'n_neighbors': list(range(1,30)),
                               'weights': ('uniform', 'distance'),
                               'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                               'p': [1, 2]
                               }
            
            ds.dataset_creation(df=True, detrended= True)
            ds.dataset_normalization(scaling)
            ds.data_summary()
            model.ds = ds
            model.verbose =3
            model.create_model()
            model.hyperparametrization()
def main():
    model_name, crypto, target = sys.argv[1].upper(), sys.argv[2].lower(), sys.argv[3].lower()
    hyperparameter_test(model_name = model_name, cryptos =[crypto], targets = [target])
if __name__ == "__main__":
    main()


    