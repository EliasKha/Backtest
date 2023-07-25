# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:28:29 2021

@author: PC
"""



import pandas as pd
import numpy as np 
import perform_strat_upon_backtest as perf_strat

path_save_backtest = 'E:/cryptos/backtesting_result/'



current_backtest = perf_strat.backtest_crypto_strategy_simple('test','test')
current_backtest.perform_backtest_start_agnostic()



def calculate_returns_on_backtest_offline(bs_filepath = path_save_backtest+'cis_zozz_17-10-2021_16_14estabaq_bt_result.xlsx'):
    """:cvar
    not in use !



    """
    df_backtest = pd.read_excel(bs_filepath)
    
    initial_usdt_capital = 10000
    
    allocation_per_trade = 0.2
