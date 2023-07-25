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

import ccxt
import datetime
import market_loader.market_loader_ccxt as market_loader
import logging
import pandas as pd
import strat_with_ewma_vol_close as strat
import xlsxwriter
logging.basicConfig(level=logging.INFO)



# ccxt get binance



import logging

from logging.handlers import RotatingFileHandler

# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger()
# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(logging.DEBUG)

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
# création d'un handler qui va rediriger une écriture du log vers
# un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
# on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
# créé précédement et on ajoute ce handler au logger
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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
def load_concat_backtest(freq='5m',start_date='1/2/2020 07:00:00',pair='BTCUSDT'):
    binance_ohlc = market_loader.deribit_algo_strat()

    now = datetime.datetime.now()
    dt_start_date = datetime.datetime.strptime(start_date,'%d/%m/%Y %H:%M:%S')
    i = dt_start_date
    
    if freq=='5m':
            dt_td = datetime.timedelta(days=3,hours=11,minutes=18)
    elif freq=='1d':
        dt_td = datetime.timedelta(days=1000)
        
    resulting_bt_df = binance_ohlc.get_calculate_close_binance(freq, i.strftime('%d/%m/%Y %H:%M:%S'))
    while i < now:
        

        df2 = binance_ohlc.get_calculate_close_binance(freq, i.strftime('%d/%m/%Y %H:%M:%S'))
        logger.info(df2.tail(2))
        resulting_bt_df = resulting_bt_df.merge(df2, how='outer').copy()
        i += dt_td

    timstmap_str = now.strftime('%d%m%Y-%H_%m')
    try:
        pass
        #resulting_bt_df.to_excel('E:/cryptos/binance_histo/'+timstmap_str+'bt_btcusdt.xlsx')
    except xlsxwriter.exceptions.FileCreateError:
        resulting_bt_df.to_excel('C:\Data_repo\crypto/'+timstmap_str+'_fq_'+str(freq)+'bt_btcusdt.xlsx')
    logger.info("saved resulting_bt_df, on : "+'E:/cryptos/binance_histo/'+timstmap_str+'bt_btcusdt.xlsx')


    return resulting_bt_df
        
def update_histo_latest_ref(file_to_update = 'E:/cryptos/binance_histo/latest_backtest_btcusdt_5m.xlsx',freq='5m'):
    
    binance_ohlc = market_loader.deribit_algo_strat()
    all_file_path_to_be_uptd = ['E:/cryptos/binance_histo/24052021-09_051dbt_btcusdt_1D.xlsx'
                                ,'E:/cryptos/binance_histo/latest_backtest_btcusdt_5m.xlsx']

    print(file_to_update)
    df_ref = pd.read_excel(file_to_update)
    df_ref.sort_values(by='date_time',inplace=True)
    last_date = df_ref['date_time'].tail(1).iloc[0]
    df_updt = load_concat_backtest(freq,datetime.datetime.strftime(last_date,'%d/%m/%Y %H:%M:%S'),'BTC/USDT')
    res = df_ref.merge(df_updt,how='outer').copy()
    try : 
        res.sort_values(by='date_time',inplace=True)
    except ValueError:
        pass
    
    res.to_excel(file_to_update,index=False)
    if freq=='5m':
        res.tail(7000).to_excel('E:/cryptos/binance_histo/latest_backtest_btcusdt_5m_short.xlsx')
    else:
        pass
    return res


def update_all_backtest_files():
    all_file_path_to_be_uptd = ['E:/cryptos/binance_histo/24052021-09_051dbt_btcusdt_1D.xlsx'
                                ,'E:/cryptos/binance_histo/latest_backtest_btcusdt_5m.xlsx']
    
    try : 
        update_histo_latest_ref(all_file_path_to_be_uptd[0],'1d')
    except FileNotFoundError:
        logging.error('not found'+all_file_path_to_be_uptd[0])
    
    try:
        update_histo_latest_ref(all_file_path_to_be_uptd[1],'5m')
    except FileNotFoundError:
        logging.error('not found'+all_file_path_to_be_uptd[1])
    
    calculate_roll_metrics_on_all_df(all_file_path_to_be_uptd[0])
    
    
    calculate_roll_metrics_on_all_df(all_file_path_to_be_uptd[1])
    calculate_roll_metrics_on_all_df('E:/cryptos/binance_histo/latest_backtest_btcusdt_5m_short.xlsx')
    
    
    
def calculate_roll_metrics_on_all_df(file_to_update = 'E:/cryptos/binance_histo/latest_backtest_btcusdt.xlsx'):
     
     df_ref = pd.read_excel(file_to_update)
     
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


if __name__ == '__main__':
    
    
    print('toto')
    
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
    