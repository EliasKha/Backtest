
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:57:22 2020

@author: SL
"""


import ccxt
import datetime
# import seaborn as sn
# import statsmodels.api as sm
# from statsmodels.regression.rolling import RollingOLS
import pandas as pd
# from pandas import DataFrame
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,7)
from datetime import datetime #, timedelta
# from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.graphics.api import qqplot
# from scipy import stats
# import statsmodels.api as sm
# from itertools import product
import warnings
warnings.filterwarnings('ignore')
# import sys
# import os
import numpy as np
# Remote Data Access
#import pandas_datareader.data as web
# reference: https://pandas-datareader.readthedocs.io/en/latest/remote_data.html
# TSA from Statsmodels
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

# Display and Plotting
import matplotlib.pylab as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
np.set_printoptions(precision=5, suppress=True) # numpy
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
# seaborn plotting style
sns.set(style='ticks', context='poster')









class another_time_series():
    """
    from : https://ccxt.readthedocs.io/en/latest/manual.html#market-data


    """

    def __init__(self):
        self.state = 'Inactive'
        self.min_window_rolling = 8
        self.rolling_window = 15

    def set_current_market(self,selected_exchange='binance'):
        if selected_exchange == 'binance':
            self.exchange_name = 'binance'
            self.exchange = ccxt.binance({
                'rateLimit': 10000,
                'enableRateLimit': True,
                # 'verbose': True,
            })
        elif selected_exchange == 'deribit':
            self.exchange_name = 'deribit'
            self.exchange = ccxt.deribit({
                'rateLimit'      : 10000,
                'enableRateLimit': True,
                # 'verbose': True,
            })


        else:
            pass

    def load_binance_ts(self, market_requested='BTC/USDT'
                        , frequency_requested='1m'
                        , from_datetime='15/02/2020 07:00:00'
                        , limit = 1200
                        ):

        """
        binance :

            1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d, 1w, 1M
        ccxt
         'timeframes': {                     // empty if the exchange.has['fetchOHLCV'] !== true
        '1m': '1minute',
        '1h': '1hour',
        '1d': '1day',
        '1M': '1month',
        '1y': '1year',

        """


        self.set_current_market()

        self.markets = self.exchange.load_markets()
        # self.from_timestamp = self.exchange.parse8601(from_datetime)
        self.dt_from_datetime = datetime.strptime(from_datetime, '%d/%m/%Y %H:%M:%S')

        self.from_timestamp = int(self.dt_from_datetime.timestamp() * 1000)

        # enable built-in rate limiting upon instantiation of the exchange
        # or switch the built-in rate-limiter on or off later after instantiation
        self.exchange.enableRateLimit = True  # enable

        candles = self.exchange.fetch_ohlcv(market_requested, frequency_requested, self.from_timestamp,limit)
        self.df_candles_binance = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_candles_binance['date_time'] = pd.to_datetime(self.df_candles_binance['Timestamp'], unit='ms')
        self.df_under_study = self.df_candles_binance.copy()

    def load_another_ts(self,requested_exchange='deribit',market_requested='BTC/USDT'
                        ,frequency_requested='5m'
                        ,from_datetime='15/02/2020 07:00:00'
                        ,limit=1200
                        ):
        self.set_current_market(requested_exchange)
        self.markets = self.exchange.load_markets()
        self.dt_from_datetime = datetime.datetime.strptime(from_datetime, '%d/%m/%Y %H:%M:%S')
        self.from_timestamp = int(self.dt_from_datetime.timestamp() * 1000)
        # enable built-in rate limiting upon instantiation of the exchange
        # or switch the built-in rate-limiter on or off later after instantiation
        self.exchange.enableRateLimit = True  # enable
        candles = self.exchange.fetch_ohlcv(market_requested, frequency_requested, self.from_timestamp,limit)
        self.df_candles_exchange = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_candles_exchange['date_time'] = pd.to_datetime(self.df_candles_exchange['Timestamp'], unit='ms')
        self.df_under_study = self.df_candles_exchange.copy()







    def return_order_book(exchange='cex', market='BTC/USDT', limit=10):
        okcoin = ccxt.okcoinusd()
        ccxt.okcoin().fetch_order_book('BTC/USD', limit)

    def zoom_in_time_axis(self, start_time='2020-02-13 00:00:00', end_time='2020-02-27 00:00:00'):
        df = self.df_candles_binance.copy()
        self.df_under_study = df[(df['date_time'] >= start_time) & (df['date_time'] <= end_time)].copy()

    def calculate_relative_returns(self, df, retcolumns):
        """calculates the normalized the returns"""
        colname = retcolumns + '_shifted'

        df[colname] = df.shift(periods=1, axis=0)[retcolumns]
        df['returns'] = df[retcolumns] - df[colname]
        df['returns_relative'] = df['returns'] / df[colname]
        df['log_returns'] = np.log(df[retcolumns] / df[colname])
        df['returns_relative'].fillna(value=0, inplace=True)
        df['log_returns'].fillna(value=0, inplace=True)
        df['ewma_rol_mean'] = df[retcolumns].rolling(window=14, min_periods=self.min_window_rolling).mean()
        df['vol_rol'] = df['log_returns'].rolling(window=self.rolling_window, min_periods=self.min_window_rolling).std()
        df['ewma_rol_mean'].fillna(method='bfill', inplace=True)
        df['returns'].fillna(method='bfill', inplace=True)
        df['vol_rol'].fillna(value=0, inplace=True)

        return df


class performance_indicators():

    def __init__(self):
        self.perf_status = 'initiated'

    def calculate_performance_indicators(self):
        pass


class deribit_algo_strat(another_time_series, performance_indicators):
    def __init__(self):
        super().__init__()
        self.state = 'OK'
        self.min_window_rolling = 5
        self.path_to_save = 'C:/PythonWorkspace/sandbox_outputs/strat_bot_cryptos/'
        self.trade_type = 'None'

    def get_calculate_close_binance(self
                                    , frequency_requested='1m'
                                    , from_datetime='20/11/2020 11:00:00'
                                    , step_size=1400
                                    ):
        """
        

        Parameters
        ----------
        frequency_requested : TYPE, optional
            DESCRIPTION. The default is '1m'.
        from_datetime : TYPE, optional
            DESCRIPTION. The default is '20/11/2020 11:00:00'.
        step_size : TYPE, optional
            DESCRIPTION. The default is 1400.
            actually , the max is 1000 ... ^^    

        Returns
        -------
        None.

        """
        self.load_binance_ts(market_requested='BTC/USDT', frequency_requested=frequency_requested
                             , from_datetime=from_datetime, limit = step_size)
        self.calculate_relative_returns(self.df_candles_binance,'Close')
        return self.df_candles_binance

    def get_calculate_close_another_market(self
                                    , requested_exchange = 'deribit'
                                    , frequency_requested='1m'
                                    , from_datetime='20/11/2020 11:00:00'
                                    , step_size=1400
                                    ):
        """


        Parameters
        ----------
        frequency_requested : TYPE, optional
            DESCRIPTION. The default is '1m'.
        from_datetime : TYPE, optional
            DESCRIPTION. The default is '20/11/2020 11:00:00'.
        step_size : TYPE, optional
            DESCRIPTION. The default is 1400.
            actually , the max is 1000 ... ^^

        Returns
        -------
        None.

        """
        self.load_another_ts(requested_exchange = requested_exchange,market_requested='BTC/USDT', frequency_requested=frequency_requested
                             , from_datetime=from_datetime, limit=step_size)
        self.calculate_relative_returns(self.df_candles_exchange, 'Close')
        return self.df_candles_exchange


# lets test a default dataset from binance , by creating an object strat
strat = deribit_algo_strat()
strat.exchange = ccxt.binance({
    'rateLimit': 10000,
    'enableRateLimit': True,
    # 'verbose': True,
})

decide_from_which_date = '20/12/2020 08:00:00'
choosen_frequency = '1m'


# strat.get_calculate_close_binance(choosen_frequency,decide_from_which_date)
# strat.df_candles_binance.columns
# strat.df_candles_binance.loc[:,('date_time','Close')].plot(x='date_time',y='Close');
# df_under_study = strat.df_candles_binance.loc[:,('date_time','Close')].copy()


def tsplot(y, lags=None, title='', figsize=(14, 8)):
    '''Examine the patterns of ACF and PACF, along with the time series plot and histogram.

    Original source: https://tomaugspurger.github.io/modern-7-timeseries.html
    '''
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    return ts_ax, acf_ax, pacf_ax

# tsplot(df_under_study.loc[:,'Close'], title='close BTC price freq:'+choosen_frequency, lags=60);

