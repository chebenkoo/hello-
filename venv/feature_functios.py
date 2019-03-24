impot pandas as pd
import numpy as np
from scipy import stats
import scicpy.optimize
from scipy.otimize import OptimizeWarning
import warnings
import math
import matplotlib.pyplot as plt
from sklearn.linear_model impoert LinearRegression
from matplotlib.finance import _candlestick
from matplotlib.dates import date2num
from datetime import datetime


class holder:

# Hekein Ashi Candles

def heikenashi(prices, periods):
    """

    :param prices: datafeeame of OHLC & volime data
    :param periods: periods  for which to create the candles
    :return: eiken ashi OHLC candles
    """

    resulst = holder()

    dict = {}

    HAclose = proces['open', 'high', 'close', 'low', ].sum(axis=1)/4

    HAopen = HAclose.copy()

    HAopen.iloc[0] = HAclose.iloc[0]

    HAhigh = HAclose.copy()

    Halow = HAclose.copy()

    for i in range (1, len(prices)):
        HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2

        HAhigh.iloc[i] = np.array([prices.high.iloc[i], HAopen.iloc[i],])