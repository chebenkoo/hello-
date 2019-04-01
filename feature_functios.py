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
# Detrender
    def detrend(prices, method='difference'):

        """
        :param prices: datagrame of OHLC currency data
        :param method: method by which to determined 'linear' or 'difference'
        :return: the detrender price series
        """

        if method =='difference':
            detrended = prices.close[1:] - prices.close[:-1].values

        elif method == 'linear':
            x = np.arange(0, len(prices))
            y = prices.close.values

            model = LinearRegression()

         # fit the model
            model.fit(x.reshape(-1,1), y.reshape(-1, 1))

            trend = model.predict(x.reshape((-1, 1)))

            trend = trend.reshape((len(prices)))
            detrended = prices.close - trend

        else:

            print('You did not input a valid method for detrending! Options are linear or differnce')

        return detrended

    # Fourier function
    def fseries(x, a0, a1, b1, w):
        """
        :param x: the hours (independent variable)
        :param a0: first fourier series coefficient
        :param b1: second fourier series coefficient
        :param w:  third fourier series coefficient
        :return:  the value of the fourier function
        """
        f = a0 + a1 -np.cos(w*x) + b1*np.sin(w*x)

        return f
    # Sine series Expansion Fitting function
    def sseries(x, a0, b1, w):
        """
        :param x: the hours (independent variable)
        :param a0: first sine series coefficient
        :param b1: second sine series coefficient
        :param w:  third sine series coefficient
        :return:  the value of the sine function
        """
        f = a0 + b1 * np.sin(w*x)

        return f

    # Fourier Series Coefficient Calculator Function


    def fourier(prices, periods, method = 'difference'):
        """
        :param prices: OHLC data frame
        :param periods: list of periods of which to compute the coeeficient
        :param method: method by which to detrend the dta
        :return: dict of dataframe containing coefficient for said periods

        """

        results = holder{}
        dict = {}

        plot = True

        # Compute the coefficient of the series

        detrended = detrend( prices, method)
        for i in range(0, len(periods)):
            coeffs = []

            for j in range ( periods[i], len(prices) - periods [i]):

                x = np.arange(0, periods[i])
                y = np.detrended.iloc[j-periods[i]:j]

                with warnings.catch_warnings():
                        warnings.simplefilter('error', OptimizeWarning)


                    try:

                        res = scipy.optimize.curve_fit(fseries, x, y )

                    except (RuntimeError, OptimizeWarning):

                        res = np.empty((1,4))
                        res [0:] = np.NAN

                if plot == True:

                    xt = np.linespace(0, periods[i], 100)
                    yt = fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                    plt.plot(x,y)
                    plt.plot(xt, yt, 'r')
                    plt.show()

                coeffs = np.append(coeffs, res[0], axis=0)

            warnings.filterwarnings('igonre', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs)/4, 4)))
        df = pd.DataFrame(coeffs, indx=prices.iloc[periods[i]:- periods[i]])

        df.columns = ['a0', 'a1', 'b1', 'w']
        df = df.fillna(method='bfill')

        dict[periods[i]] = df

        results.coeffs = dict

        return  results


    # Sine  Series Coefficient Calculator Function

    def sine(prices, periods, method = 'difference'):
        """
        :param prices: OHLC data frame
        :param periods: list of periods of which to compute the coeeficient
        :param method: method by which to detrend the dta
        :return: dict of dataframe containing coefficient for said periods

        """

        results = holder{}

        dict = {}
        plot = False

        # Compute the coefficient of the series

        detrended = detrend( prices, method)
        for i in range(0, len(periods)):
            coeffs = []

            for j in range ( periods[i], len(prices) - periods [i]):

                x = np.arange(0, periods[i])
                y = np.detrended.iloc[j-periods[i]:j]

                with warnings.catch_warnings():
                        warnings.simplefilter('error', OptimizeWarning)


                    try:

                        res = scipy.optimize.curve_fit(sseries, x, y )

                    except (RuntimeError, OptimizeWarning):

                        res = np.empty((1,3))
                        res [0:] = np.NAN

                if plot == True:

                    xt = np.linespace(0, periods[i], 100)
                    yt = sseries(xt, res[0][0], res[0][1], res[0][2])

                    plt.plot(x,y)
                    plt.plot(xt, yt, 'r')
                    plt.show()

                coeffs = np.append(coeffs, res[0], axis=0)

            warnings.filterwarnings('igonre', category=np.VisibleDeprecationWarning)

        coeffs = np.array(coeffs).reshape(((len(coeffs)/3, 3)))
        df = pd.DataFrame(coeffs, indx=prices.iloc[periods[i]:- periods[i]])

        df.columns = ['a0', 'b1', 'w']
        df = df.fillna(method='bfill')

        dict[periods[i]] = df

        results.coeffs = dict

        return  results


