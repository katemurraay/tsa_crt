from models.dl.lstm_gru_hybrid import LSTM_GRU
from util import plot_training, save_results, dataset_binance, r2
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, median_absolute_error, explained_variance_score, mean_squared_log_error
import numpy as np
import pandas as pd

def main():
    h=0
    targets = ['close']
    cryptos =  ['btc',  'eth', 'ltc', 'xmr', 'xrp']
    retrain = [0, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30]
    outputs =[30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31] 
    scaling = ['minmax']
    tuned =  0
    window = 30
    for t in targets:
        for c in cryptos:
            add_split_value = 0
            mse, rmse, mape, r2_score, mae = [], [], [], [], []
            for index, r in enumerate(retrain):
                output = outputs[index]
                
                experiment_name = 'hybrid-' + c + '-' + t + '-h' + str(h) + '-'+ str(index) +'m'
                ds = dataset_binance.BinanceDataset(filename='crypto_task_' + c + '.csv', input_window=window, output_window=1,
                                                            horizon=h, training_features=['close'],
                                                            target_name=['close'], train_split_factor=0.8)
                df, diff_df = ds.differenced_dataset()
                ds.df = diff_df
                if index > 0:
                    add_split_value += r
                
                ds.add_split_value = add_split_value
                
                if tuned:
                    parameters = pd.read_csv("param/p_HYBRID-"+t+ '-'+c +"-w30-h0.csv").iloc[0] 
                    p  = {  'first_lstm_dim': parameters['first_lstm_dim'],
                            'lstm_activation': parameters['lstm_activation'],
                            'first_dropout_rate': parameters['first_dropout_rate'], 
                            'second_lstm_dim': parameters['second_lstm_dim'],
                            'first_dense_dim':  parameters['first_dense_dim'],
                            'dense_activation': parameters['dense_activation'],
                            'dense_kernel_init':parameters['dense_kernel_init'],
                            'gru_dim':parameters['gru_dim'],
                            'gru_activation':parameters['gru_activation'],
                            'second_dropout_rate':  parameters['second_dropout_rate'],
                            'second_dense_dim': parameters['second_dense_dim'],
                            'batch_size': parameters['batch_size'],
                            'epochs': parameters['epochs'],
                            'patience': parameters['patience'],
                            'optimizer': parameters['optimizer'],
                            'lr': parameters['lr'],
                            'momentum': parameters['momentum'],
                            'decay': parameters['decay'],
                            
                            }
                    
                else:
                    p = {
                        'first_lstm_dim': 75,
                        'lstm_activation': 'relu',
                        'first_dropout_rate': 0.05, 
                        'second_lstm_dim':  50,
                        'first_dense_dim':  32,
                        'dense_activation': 'relu',
                        'dense_kernel_init': 'he_normal',
                        'gru_dim':50,
                        'gru_activation':'relu',
                        'second_dropout_rate':  0.0,
                        'second_dense_dim':  64,
                        'batch_size': 256,
                        'epochs': 200,
                        'patience': 50,
                        'optimizer': 'adam',
                        'lr': 1E-3,
                        'momentum': 0.9,
                        'decay': 1E-4,
                        }
                model =LSTM_GRU(experiment_name)
                model.ds = ds 
                ds.dataset_creation(df=True, detrended= True)
                ds.dataset_normalization(scaling)
                ds.data_summary()
                to_predict = ds.X_test[:output]
                yhat, train_model = model.training(p, X_test=to_predict)                                                                 
                preds = np.array(yhat).reshape(-1, 1)
                np_preds = ds.inverse_transform_predictions(preds = preds)
                inversed_preds = ds.inverse_differenced_dataset(diff_vals= np_preds, df=df, l = (len(ds.y_test_array)))
                ds.df = df
                ds.dataset_creation(df=True)
                labels = ds.y_test_array[(h):(len(inversed_preds)+h)].reshape(-1, 1)
                ds.add_split_value = 0
                ds.df = df
                ds.dataset_creation(df=True)
                ds.dataset_normalization(scaling)
                n_preds = ds.scale_predictions(preds= inversed_preds)                               
                n_labels =  ds.scale_predictions(preds= labels)
                mse.append(mean_squared_error(n_labels, n_preds))
                rmse.append(np.sqrt(mean_squared_error(n_labels, n_preds)))
                mae.append(mean_absolute_error(n_labels, n_preds))
                mape.append(mean_absolute_percentage_error(n_labels, n_preds))
                r2_score.append(r2.r_squared(n_labels, n_preds))
                print("MSE",mean_squared_error(n_labels, n_preds))
                print("MAE", mean_absolute_error(n_labels, n_preds))
                print("MAPE", mean_absolute_percentage_error(n_labels, n_preds))
                print("RMSE",np.sqrt(mean_squared_error(n_labels, n_preds)))
                print("R2", r2.r_squared(n_labels, n_preds))
                n_experiment_name = experiment_name + '_N'
                save_results.save_output_csv(preds = n_preds, labels= n_labels, feature=t, filename= n_experiment_name, bivariate=len(ds.target_name) > 1)
            if not tuned:
                save_results.save_params_csv(model.p, model.name)      
               
            save_results.save_metrics_csv(mses = mse, maes= mae, rmses= rmse, mapes=mape, filename=experiment_name, r2=r2_score)
                                                  

if __name__ == "__main__":
    main()