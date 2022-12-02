# On Forecasting Cryptocurrency Prices: a Comparison of Machine Learning, Deep Learning and Ensembles

## Introduction

Traders and investors are interested in accurately predicting cryptocurrency prices to increase returns and minimize risk. However, due to their uncertainty, volatility, and dynamism, forecasting crypto prices is a challenging time-series analysis task. Researchers have proposed predictors based on statistical, machine learning (ML) and deep learning (DL) approaches, but the literature: is narrow because it focuses on predicting only the prices of the few most-famous cryptos; it is scattered because it compares different models on different cryptos inconsistently, and it lacks generality because solutions are overly complex and hard to reproduce in practice. The main goal of this paper is to provide a comparison framework that overcomes these limitations. We use this framework to run extensive experiments where we compare the performances of widely-used statistical, ML, and DL approaches in the literature for predicting the price of five popular cryptocurrencies, i.e.\ XRP, Bitcoin (BTC), Litecoin (LTC), Ethereum (ETH) and Monero (XMR). To the best of our knowledge, we are also the first to propose using the Temporal Fusion Transformer (TFT) on this task. Moreover, we extend our investigation to hybrid models and ensembles to assess whether combining single models boosts prediction accuracy. Our evaluation shows that DL approaches are the best predictors, particularly the LSTM, and this is consistently true across all the cryptos examined. To ensure reproducibility and stimulate future research contribution, we share the dataset and the code of the experiments.

## Python Dependencies
* arch                      5.1.0
* keras                     2.8.0
* matplotlib                3.3.4
* numpy                     1.21.5
* pandas                    1.2.3
* python                    3.7.9
* statsmodels               0.12.2
* talos                     1.0.2 
* tensorflow                2.8.0
* tensorflow-gpu            2.8.0


## Project Structure
* **talos**: contains for each deep learning model the list of optimal hyperparameters found with Talos.
* **img**: contains output plot for predictions, models and loss function.
* **models**: contains the definition of statistical and deep learning models. One can Test the model from scratch using the optimal parameters found with Talos, look for the optimal hyperparameters by changing the search space dictionary or load a saved model and make new forecasts.
* **param**: contains for each statistical model the list of optimal parameters found.
* **res**: contains the results of the prediction.
* **saved_data**: contains the preprocessed datasets.
* **saved_models**: contains the model saved during the test  phase.
* **time**: contains measurements of the time for test , fine-tuning and inference. phases.
* **util**: contains useful methods for initialising the datasets, plotting and saving the results.

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


