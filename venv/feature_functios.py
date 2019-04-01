import pandas as pd
import numpy as np
from scipy import stats
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime


class holder:

# Hekein Ashi Candles

    def heikenashi(prices, periods):
        """

        :param prices: datafeeame of OHLC & volime data
        :param periods: periods  for which to create the candles
        :return: heiken ashi OHLC candles
        """


        results = holder()

        dict = {}

        HAclose = prices[['open', 'high', 'close', 'low']].sum(axis=1) / 4

        HAopen = HAclose.copy()

        HAopen.iloc[0] = HAclose.iloc[0]

        HAhigh = HAclose.copy()

        HAlow = HAclose.copy()

        for i in range(1, len(prices)):
            HAopen.iloc[i] = (HAopen.iloc[i - 1] + HAclose.iloc[i - 1]) / 2

            HAhigh.iloc[i] = np.array([prices.high.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).max()

            HAlow.iloc[i] = np.array([prices.low.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).min()

        df = pd.concat((HAopen, HAhigh, HAlow, HAclose), axis=1)
        df.columns = [['open', 'high', 'close', 'low']]

# df.index = df.index.dropleve(0)

        dict[periods[0]] = df

        results.candles = dict

        return results
