import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
"""
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

import sklearn.metrics



FSLR_input = pd.read_csv('FSLR.csv', index_col='Дата')
"""

tickers = np.array(['ARKW', 'AAPL', 'T', 'NOK', 'FSLR', 'SWKS', 'GDX', 'PYPL', 'ITA'])

print(tickers)
tickers_history_prices = pd.DataFrame()

for ticker in tickers:
    ticker_sheet = pd.read_excel('shares_history.xlsx', index_col='Дата', sheetname=ticker)
    ticker_price = ticker_sheet["Цена"]
    ticker_price.name = ticker

    tickers_history_prices[ticker]=ticker_price
    """
    if (tickers_history_prices is None):
        tickers_history_prices = ticker_price
        tickers_history_prices = tickers_history_prices.to_frame()
    else:
        pd.concat([tickers_history_prices, ticker_price.to_frame()], axis=1)
"""
print (tickers_history_prices.head(10))

print(tickers_history_prices.corr())
"""
FSLR_input = pd.read_excel('shares_history.xlsx', index_col='Дата', sheetname='FSLR')
FSLR_price = FSLR_input["Цена"]
FSLR_price.name ="FSLR"

ARKW_input = pd.read_excel('shares_history.xlsx', index_col='Дата', sheetname='ARKW')
ARKW_price = ARKW_input["Цена"]
ARKW_price.name ="ARKW"

AAPL_input = pd.read_excel('shares_history.xlsx', index_col='Дата', sheetname='AAPL')
AAPL_price = AAPL_input["Цена"]
AAPL_price.name ="AAPL"
"""


"""
print(ARKW_price)
FSLR_price.index = pd.to_datetime(FSLR_price.index)
"""
"""
corr = np.corrcoef(ARKW_price, AAPL_price)

print("Correlation: ", corr)


plt.plot(FSLR_price)
plt.plot(ARKW_price)
plt.plot(AAPL_price)
# We still need to add the axis labels and title ourselves
plt.title("Shares Prices")
plt.ylabel("Price")
plt.xlabel("Date")

plt.show()


"""