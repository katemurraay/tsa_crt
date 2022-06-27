from util.dataset import DatasetInterface
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from binance.client import Client
from binance import BinanceSocketManager
import pickle


class BinanceDataset(DatasetInterface):
    def __init__(self,  filename="", input_window=10, output_window=1, horizon=0, training_features=[], target_name=[],
                 train_split_factor=0.8, apiKey = "", apiSecurity = ""):
        """
        Call to the parent constructor [DatasetInterface] and passing the required parameters:
            string: filename
            int: input_window
            int: output_window
            int: horizon
            array: training features
            array: target name
            float: train test split factor
        """
        super().__init__(filename, input_window, output_window, horizon, training_features, target_name, train_split_factor)
        """string: api key required for binance api access"""
        self.apiKey = apiKey
        """string: api security required for binance api access """
        self.apiSecurity = apiSecurity
        
    
    """
    Builds DataFrame to structure data returned from Binance API
    """
    def create_frame(self, data):
        df = pd.DataFrame(data)
        df = df.iloc[:,0:9]
        """columns returned by binance """
        df.columns = ['OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumTrades']
        
        """float: open price"""
        df.Open = df.Open.astype(float)
        """float: close price"""
        df.Close = df.Close.astype(float)
        """float: high price"""
        df.High = df.High.astype(float)
        """float: low price"""
        df.Low= df.Low.astype(float)
        """float: volume traded"""
        df.Volume= df.Volume.astype(float)
        """float: asset volume quoted"""
        df.QuoteAssetVolume= df.QuoteAssetVolume.astype(float)
        """int: number of trades"""
        df.NumTrades = df.NumTrades.astype(int)
        """Date: open time"""
        df.OpenTime = pd.to_datetime(df.OpenTime, unit = 'ms')
        """Date: close time"""
        df.CloseTime = pd.to_datetime(df.CloseTime, unit = 'ms')
        return df
    
    
    """
    Retrieves Data from Binance API given the symbol of the cryptocurrency, a start date and an end date 
    """
    def get_binance_data(self, sym, start_date, end_date):
       try:
            client = Client(self.apiKey, self.apiSecurity)
            print("Logged in")
            interval = Client.KLINE_INTERVAL_1DAY
            klines = client.get_historical_klines(sym, interval, start_date, end_date)
            df = self.create_frame(klines)
            df_b = df.iloc[:, 0:5]
            return df_b
       except: 
            print('Invalid Login')
      
    """
    Saves DataFrame to .csv file witin the saved_data folder
    """
    def save_to_csv(self, df):
        df.columns = ['date', 'open', 'high', 'low', 'close']
        save_name = self.data_path + self.data_file
        df.to_csv(save_name)

    """
    Combines Data from Investing.com [stored in Github Repository] with Binance API Data 
    """
    def build_crypto_dataset(self, name, year1, year2, sym, start_date, end_date):
        # get cryptocurrency csv file from github (downloaded from: Investing.com)
        path ='https://github.com/katemurraay/TFT_Data/blob/main/'+name +'_'+year1+'_'+ year2 +'.csv?raw=true'
        df_git = pd.read_csv(path)
        df_git.Date = pd.to_datetime(df_git.Date, format='%d/%m/%Y')
        # get binance data for cryptocurrency
        df_binance = self.get_binance_data(sym, start_date, end_date)
        #restructure datasets to match one another
        column_names = ["Date", "Open", "High", "Low", "Close"]
        df_git = df_git.reindex(columns=column_names)
        df_binance = df_binance.rename(columns = {'OpenTime':'Date'})
        #get index of binance start date in the github dataframe
        i = df_git.index[df_git['Date'] == df_binance['Date'].iloc[0]].tolist()
        #combine both datasets into a list
        df_a = df_git[:i[0]]
        df_b = df_binance.iloc[:, 0:5]
        list_combine = []
        list_combine.extend(df_a.values)
        list_combine.extend(df_b.values)
        #convert list to a dataframe
        df_combined = pd.DataFrame(list_combine, columns=['date', 'open', 'high', 'low', 'close'])
        self.save_to_csv(df= df_combined)
  
