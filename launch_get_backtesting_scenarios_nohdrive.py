# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri 27 03 07:47:45 2021

@author: SL
"""
"""
lets imagine a benchmark for parameters : 

    Run bench mark period wth a given param (Jul-20,Aug-20 , Sept-20 )

    consolidate results => send recap of results 
    name the scenar after the param value

    come up with a set of optimised param 
    param are : 
        - threshold inflex
        - threshold slope
        - threshold vol 
    do an histogram of the past 
"""

# imports
import yfinance as yf
import numpy as np

import ccxt
import datetime
import market_loader.market_loader_ccxt as market_loader
import logging
import pandas as pd
import strat_with_ewma_vol_close as strat
import xlsxwriter
from logging.handlers import RotatingFileHandler
from datetime import timedelta




logging.basicConfig(level=logging.INFO)
# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger()

# on met le niveau du logger à DEBUG, comme ça il écrit tout
# logger.setLevel(logging.DEBUG)

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

# création d'un handler qui va rediriger une écriture du log vers
# un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
# file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)

# on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
# créé précédement et on ajoute ce handler au logger
# file_handler.setLevel(logging.ERROR)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# création d'un second handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
logger.addHandler(stream_handler)

# Après 3 heures, on peut enfin logguer
# Il est temps de spammer votre code avec des logs partout :
logger.info('Hello')
logger.warning('Testing %s', 'foo')






binance_ohlc = market_loader.deribit_algo_strat()
"""
binance_ohlc.get_calculate_close_binance('5m','20/3/2020 07:00:00')
print('first date is :',binance_ohlc.df_candles_binance.loc[:,'date_time'][0])
print('last date is :',binance_ohlc.df_candles_binance.loc[:,'date_time'][-1:])

print('timedelta :',binance_ohlc.df_candles_binance.loc[:,'date_time'][-1:] - binance_ohlc.df_candles_binance.loc[:,'date_time'][0])

dt_td = datetime.timedelta(days=3,hours=11,minutes=18)

print('calculated date is :', binance_ohlc.df_candles_binance.loc[:,'date_time'][0]+dt_td)

"""
def load_concat_backtest(freq='5m',start_date='12/11/2021 07:00:00',pair='BTCUSDT'):
    """
    """

    binance_ohlc = market_loader.deribit_algo_strat()
    now = datetime.datetime.now()
    dt_start_date = datetime.datetime.strptime(start_date,'%d/%m/%Y %H:%M:%S')
    i = dt_start_date

    match freq:
        case '5m':
            dt_td = datetime.timedelta(days=3,hours=11,minutes=18)
        case '1d':
            dt_td = datetime.timedelta(days=1000)
        case'15m':
            dt_td = datetime.timedelta(hours=250)
        case'1m':
            dt_td = datetime.timedelta(minutes=999)
        
    resulting_bt_df = binance_ohlc.get_calculate_close_binance(freq, i.strftime('%d/%m/%Y %H:%M:%S'))

    while i < now:
        df2 = binance_ohlc.get_calculate_close_binance(freq, i.strftime('%d/%m/%Y %H:%M:%S'))
        logger.info(df2.tail(2))
        resulting_bt_df = resulting_bt_df.merge(df2, how='outer').copy()
        i += dt_td

    timstmap_str = now.strftime('%d%m%Y-%H_%m')

    try:
        resulting_bt_df.to_excel('C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat\historical_rates/05082021-18_08_fq_15mbt_btcusdt.xlsx')
    except xlsxwriter.exceptions.FileCreateError:
        resulting_bt_df.to_excel('C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat/Data_repo/crypto/'+timstmap_str+'_fq_'+str(freq)+'bt_btcusdt.xlsx')

    logger.info("saved resulting_bt_df, on : "+'C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat\historical_rates/'+timstmap_str+'bt_btcusdt.xlsx')

    return resulting_bt_df
        
def update_histo_latest_ref(file_to_update = 'E:/cryptos/binance_histo/latest_backtest_btcusdt_5m.xlsx',freq='5m'):
    """
    """

    print('FILE: ' + file_to_update)
    df_ref = pd.read_excel(file_to_update)
    df_ref.sort_values(by='date_time',inplace=True)
    last_date = df_ref['date_time'].tail(1).iloc[0]
    df_updt = load_concat_backtest(freq,datetime.datetime.strftime(last_date,'%d/%m/%Y %H:%M:%S'),'BTC/USDT')
    res = df_ref.merge(df_updt,how='outer').copy()
    
    try : 
        res.sort_values(by='date_time',inplace=True)
    except ValueError:
        pass
    df_ref = add_close_prices(df_ref, frequency=freq, contract_name='ETH-USD')
    df_ref = add_close_prices(df_ref, frequency=freq, contract_name='SPY')
    df_ref.to_excel(file_to_update,index=False)


    
    



def add_close_prices(res, frequency, contract_name):
    print("ADD CLOSE PRICE: " + contract_name)

    #Set up variables
    start_date = res['date_time'].min()
    end_date = res['date_time'].max() 
    match frequency:
        case '1d':
            end_date_add = timedelta(days=1)
            freq_str = '1d'
        case '15m':
            end_date_add = timedelta(minutes=15)
            freq_str = '15T'
        case '5m':
            end_date_add = timedelta(minutes=5)
            freq_str = '5T'
        case '1m':
            end_date_add = timedelta(minutes=1)
            freq_str = 'T'

    #Download data using yfinance
    contract = yf.Ticker(contract_name)
    historical_data = contract.history(start=start_date, end=end_date + end_date_add, interval=frequency)  
    #When no trades at all in the period, needs improvment
    if historical_data['Close'].empty:
        print('No trades at all')
        ## Get the latest date with closing price
        close_index = historical_data.columns.get_loc('Close')
        lates_closing_price = latest_data_point(start_date, frequency, contract)
        zeros_historical_data = pd.DataFrame(np.full((len(res['date_time']), len(historical_data.columns)), lates_closing_price), columns=historical_data.columns)
        historical_data = pd.concat([historical_data.iloc[:close_index], zeros_historical_data, historical_data.iloc[close_index:]]).reset_index(drop=True)

    else:
        #Complete the last data when some dates are missing (weekends)
        complete_dates = pd.date_range(start=start_date, end=end_date, freq=freq_str)
        complete_df = pd.DataFrame(index=complete_dates)
        complete_df = complete_df.tz_localize(historical_data.index.tz)
        merged_df = pd.merge(complete_df, historical_data, left_index=True, right_index=True, how='left') 

        for pos, close_price in enumerate(merged_df['Close']):
            if pd.isna(close_price):
                if pos != 0: 
                    merged_df['Close'][pos] = merged_df['Close'][pos-1]
                else: #When the 1st element doesnt have any closing price, need improvment
                    ## Get the latest date with closing price 
                    lates_closing_price = latest_data_point(start_date, frequency, contract)
                    merged_df['Close'][pos] = lates_closing_price

        historical_data = merged_df

    # Add the data to the Dataframe 
    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data.index = historical_data.index.strftime('%Y-%m-%d %H:%M:%S') 
    try: 
        close_prices = historical_data['Close'].values
        res[f'{contract_name} Close'] = close_prices
    except:
        logging.error('Data not clean')
    
    return res

def latest_data_point(date, frequency, contract):
    print('its doing something')
    historical_data = contract.history(start=date - timedelta(weeks=1), end=date, interval=frequency)  
    return historical_data['Close'][-1]



def update_all_backtest_files(): 
    all_file_path_to_be_uptd = [
                                'C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat/test_files/test_1d.xlsx',
                                'C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat/test_files/test_1m.xlsx',
                                'C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat/test_files/test_5m.xlsx',
                                'C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat/test_files/test_15m.xlsx',
                                ]
    

    try:
        update_histo_latest_ref(all_file_path_to_be_uptd[0],'1d')
    except FileNotFoundError:
        logging.error('not found'+all_file_path_to_be_uptd[0])

    try:
        update_histo_latest_ref(all_file_path_to_be_uptd[1],'1m')
    except FileNotFoundError:
        logging.error('not found'+all_file_path_to_be_uptd[1])
    
    try:
        update_histo_latest_ref(all_file_path_to_be_uptd[2],'5m')
    except FileNotFoundError:
        logging.error('not found'+all_file_path_to_be_uptd[2])
    
    try:
        update_histo_latest_ref(all_file_path_to_be_uptd[3],'15m')
    except FileNotFoundError:
        logging.error('not found'+all_file_path_to_be_uptd[3])













    try:
        pass
        # calculate_roll_metrics_on_all_df(all_file_path_to_be_uptd[3])
    except FileNotFoundError:
         logging.error('no metrics calculated'+ all_file_path_to_be_uptd[3])

    try:
        pass
        # calculate_roll_metrics_on_all_df(all_file_path_to_be_uptd[0])
    except FileNotFoundError:
         logging.error('no metrics calculated'+ all_file_path_to_be_uptd[0])
    
    try:
        pass
        # calculate_roll_metrics_on_all_df(all_file_path_to_be_uptd[1])
    except FileNotFoundError:
          logging.error('no metrics calculated'+ all_file_path_to_be_uptd[1])
    
    try: 
        pass
        # calculate_roll_metrics_on_all_df('E:/cryptos/binance_histo/latest_backtest_btcusdt_5m_short.xlsx')
    except FileNotFoundError:
          logging.error('no metrics calculated'+'E:/cryptos/binance_histo/latest_backtest_btcusdt_5m_short.xlsx')
    
    # calculate_roll_metrics_on_all_df(all_file_path_to_be_uptd[2])
    

    
    
    
    
def calculate_roll_metrics_on_all_df(file_to_update = 'E:/cryptos/binance_histo/latest_backtest_btcusdt.xlsx'):
         
    try:
        df_ref = pd.read_excel(file_to_update)
    except FileNotFoundError:
       return('no file')
    
    obj_test = strat.deribit_algo_strat()
    df_metrics = obj_test.calculate_rolling_metrics(df_ref,'Close')
     
    try:
        df_metrics.to_excel(file_to_update,index=False)
        logging.info('metrics calculated on '+file_to_update)
    except FileNotFoundError:
        logging.error('no file found on '+file_to_update)
     
     
 

       


#df_final = load_concat_backtest()


# def including frequency , market , ceiling , BTC ou ETH ?

# while  outer loop , stops when date > ceiling


# global stat max local vol , min local vol , histogram ...


def download_btc_data():
    start_date = "2023-07-03"
    # end_date = start_date + timedelta(minutes=2)
    ticker = "BTC-USD"
    interval = "1d"

    start_date =  pd.to_datetime(start_date)
    end_date = start_date #+ pd.DateOffset(days=1)

    btc_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    print(btc_data)
    # btc_data.index = btc_data.index.tz_localize(None)
    # btc_data.to_excel(r'C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat/test_files/test_1m.xlsx',header = True, index = True)






if __name__ == '__main__':
    
    
    print('toto')
    # download_btc_data()
    update_all_backtest_files()




    #res = update_histo_latest_ref()
    #res.to_clipboard()
    
    #freq = '2h'
    #start_date = '1/2/2017 07:00:00'
    #from_date_boundary = datetime.datetime(2017,2,23)

    
    """
    pair = 'BTCUSDT'


    now = datetime.datetime.now()
    dt_start_date = datetime.datetime.strptime(start_date, '%d/%m/%Y %H:%M:%S')
    i = dt_start_date
    
    #◄ if daily , dt_td = 1000 days 
    # dt_td = datetime.timedelta(days=997)
    #if 5m, 
    #dt_td = datetime.timedelta(days=3, hours=11, minutes=18)
    #if 2h, 
    dt_td = datetime.timedelta(days=83)
    
    
    
    resulting_bt_df = binance_ohlc.get_calculate_close_binance(freq, from_date_boundary.strftime('%d/%m/%Y %H:%M:%S'))
    
    while i < now:
        i += dt_td

        df2 = binance_ohlc.get_calculate_close_binance(freq, i.strftime('%d/%m/%Y %H:%M:%S'))
        print(i)
        print(df2.tail(2))
        resulting_bt_df = resulting_bt_df.merge(df2, how='outer').copy()

    timstmap_str = now.strftime('%d%m%Y-%H_%m')
    
    # need to delete the doubloons !! 
    
    resulting_bt_df.to_excel('E:/cryptos/binance_histo/' + timstmap_str +freq+'bt_btcusdt_2h.xlsx')
    """
    
#C:/Users/elias/OneDrive/Bureau/Quanta_Analytics/backtesting_benchmark_vaqat/cryptos/binance_histo/latest_backtest_btcusdt.xlsx
#E:/cryptos/binance_histo/latest_backtest_btcusdt_5m_short.xlsx