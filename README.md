# On Forecasting Cryptocurrency Prices: a Comparison of Machine Learning, Deep Learning and Ensembles

## Introduction

Traders and investors are interested in accurately predicting cryptocurrency prices to increase returns and minimize risk. However, due to their uncertainty, volatility, and dynamism, forecasting crypto prices is a challenging time series analysis task. {Researchers have proposed predictors based on statistical, machine learning (ML), and deep learning (DL) approaches, but the literature is limited. Indeed, it is narrow because it focuses on predicting only the prices of the few most famous cryptos. In addition, it is scattered because it compares different models on different cryptos inconsistently, and it lacks generality because solutions are overly complex and hard to reproduce in practice. The main goal of this paper is to provide a comparison framework that overcomes these limitations. We use this framework to run extensive experiments where we compare the performances of widely used statistical, ML, and DL approaches in the literature for predicting the price of five popular cryptocurrencies, i.e.,\ XRP, Bitcoin (BTC), Litecoin (LTC), Ethereum (ETH), and Monero (XMR). To the best of our knowledge, we are also the first to propose using the temporal fusion transformer (TFT) on this task. Moreover, we extend our investigation to hybrid models and ensembles to assess whether combining single models boosts prediction accuracy. Our evaluation shows that DL approaches are the best predictors, particularly the LSTM, and this is consistently true across all the cryptos examined. {LSTM reaches an average RMSE of $0.0222$ and MAE of $0.0173$, respectively, $2.7\%$ and $1.7\%$ better than the second-best model}. To ensure reproducibility and stimulate future research contribution, we share the dataset and the code of the experiments.

## Python Dependencies
* arch                      5.1.0
* keras                     2.8.0
* matplotlib                3.3.4
* numpy                     1.21.5
* pandas                    1.2.3
* python                    3.7.9
* statsmodels               0.13.2
* talos                     1.0.2 
* tensorflow                2.8.0
* tensorflow-gpu            2.8.0
* torch                     1.11.0           
* scikit-learn              1.0.2  
* pytorch-lightning         1.5.10
* python-binance            1.0.16
* optuna                    2.10.1 
* darts                     0.20.0 
* sklearn                   0.0


## Project Structure
* **talos**: contains the list of optimal hyperparameters found with Talos for each deep model.
* **models**: contains the definition of statistical, machine learning and deep learning models. One can Test the model from scratch using the optimal parameters found with Talos, look for the optimal hyperparameters by changing the search space dictionary or load a saved model and make new forecasts.
* **param**: contains the list of optimal parameters found for every model.
* **res**: contains the results of the prediction.
* **saved_data**: contains the datasets.
* **saved_models**: contains the model saved when fitting the model.
* **util**: contains useful methods for initialising the datasets, plotting, calculating and saving the results.

## Statistical Methods

#### Test ARIMA

```bash
python arima_test.py
```

## Machine Learning Methods
#### Test SVR

```bash
python svr_test.py
```
#### Test RF

```bash
python rf_test.py
```
#### Test kNN

```bash
python knn_test.py
```
## Deep Learning Methods

#### Test LSTM

```bash
python lstm_test.py
```
#### Test GRU

```bash
python gru_test.py
```

#### Test LSTM-GRU Hybrid

```bash
python hybrid_test.py
```

#### Test TFT

```bash
python tft_test.py
```

#### Test TCN

```bash
python tcn_test.py
```


