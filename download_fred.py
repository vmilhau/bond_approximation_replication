from fredapi import Fred
import pandas as pd

fred = Fred(api_key='6a032aece57270ae2a352eaf8020890c')

tickers = ['DGS1', 'DGS10', 'T10YFF']
names = ['1 Year', '10 Years', 'Spread']

for i, j in zip(tickers, names):
    fred.get_series(i).rename(j).to_csv(f'{i}.csv')
