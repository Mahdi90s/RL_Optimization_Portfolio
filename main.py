import yfinance as yf
import pandas as pd
from IPython.display import display


assets = ['NVDA', 'AAPL', 'AMZN', 'JPM', 'IBM', 'MSFT', 'TSLA', 'GOOGL', 'META', 'HSBC']
start_date ='2019-01-01'
end_date = '2025-01-01'


# # with pd.option_context('display.max_row', 30, 'display.max_columns', 5):
# #     display(df)
    
# # print (df.info())
# # print (df.describe())
# # print (df.dtypes)
# # print (df.isna().sum())
# # print(df.index.to_series().min())
# # print(df.index.to_series().max())

for symbol in assets:
    df = yf.download(tickers=symbol, start= start_date , end= end_date, interval='1d')
    df.to_csv(f'/home/micheal/Documents/Python_Library/RL_Optimization_Portfolio/data/raw/{symbol}_daily.CSV')

df = pd.read_csv('/home/micheal/Documents/Python_Library/RL_Optimization_Portfolio/data/raw/AAPL_daily.CSV')

print(df.head())                    