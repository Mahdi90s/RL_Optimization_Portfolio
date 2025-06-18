import yfinance as yf
import pandas as pd
from IPython.display import display
from scipy.stats import zscore
import numpy as np


# assets = ['NVDA', 'AAPL', 'AMZN', 'JPM', 'IBM', 'MSFT', 'TSLA', 'GOOGL', 'META', 'HSBC']
# start_date ='2019-01-01'
# end_date = '2025-01-01'

# df = yf.download(tickers=assets, start= start_date , end= end_date, interval='1d')

# sta = df.stack()

# df_flt = sta.reset_index()
# is_multi_index = isinstance(df_flt.index, pd.MultiIndex)  
# is_multi_col= isinstance(df_flt.columns, pd.MultiIndex)

# print (df_flt.head())
# print (df_flt.info())
# print (df_flt.describe())
# print (df_flt.dtypes)
# print (df_flt.isna().sum())
# print(df_flt.index.to_series().min())
# print(df_flt.index.to_series().max())
# print ('Row', is_multi_index)
# print ('col', is_multi_col)


# for symbol in assets:
#     df = yf.download(tickers=symbol, start= start_date , end= end_date, interval='1d')
#     sta = df.stack()
#     df_flt = sta.reset_index()
#     df_flt.to_csv(f'/home/micheal/Documents/Python_Library/RL_Optimization_Portfolio/data/raw/{symbol}_daily.CSV')

#_______________________________________________________________________________________________________________________________________________________________________________________________

# appl = pd.read_csv('/home/micheal/Documents/Python_Library/RL_Optimization_Portfolio/data/raw/AAPL_daily.CSV', parse_dates=["Date"], index_col="Date")
# msft = pd.read_csv('/home/micheal/Documents/Python_Library/RL_Optimization_Portfolio/data/raw/MSFT_daily.CSV', parse_dates=["Date"], index_col="Date")


# df = pd.concat([appl, msft])


# is_multi_index = isinstance(df.index, pd.MultiIndex)  
# is_multi_col= isinstance(df.columns, pd.MultiIndex)

# df = df.drop(columns=['Unnamed: 0']).copy()

# print (df.head())
# print (df.columns)
# print (df[['Ticker']])
# print (df.info())
# print (df.describe())
# print (df.dtypes)
# print (df.isna().sum())
# print (df.index.to_series().min())
# print (df.index.to_series().max())
# print ('Row', is_multi_index)
# print ('col', is_multi_col)


# df['LogReturn'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))

# df.dropna(subset=['LogReturn'], inplace=True)

# df['LogReturn_Z'] = df.groupby('Ticker')['LogReturn'].transform(lambda x: (x - x.mean())/ x.std())

# print (df.head())
# print (df.isna().sum())
# print (df[df.isna().any(axis=1)])
# print (df.tail())

# df.to_csv('/home/micheal/Documents/Python_Library/RL_Optimization_Portfolio/data/processed/AP_MS_daily.CSV')

#_______________________________________________________________________________________________________________________________________________________________________________________________

addres = '/home/micheal/Documents/Python_Library/RL_Optimization_Portfolio/data/processed/AP_MS_daily.CSV'

df = pd.read_csv(addres, parse_dates=['Date'], index_col='Date')

print (df.head())
print (df.columns)