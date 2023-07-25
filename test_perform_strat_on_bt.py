# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:24:30 2021

@author: PC
"""
import pandas as pd
import numpy as np 
import perform_strat_upon_backtest as perf_strat


"""
obj_test.load_backtest_scenario()
obj_test.measure_rolling_ewma_std('test')

print(obj_test.df_backtest.tail())

obj_test.perform_backtest()


obj_test.df_backtest.to_clipboard()





current_backtest = perf_strat.backtest_crypto_strategy_simple('test','test')
current_backtest.perform_backtest_start_agnostic()

"""






current_backtest = perf_strat.backtest_crypto_strategy_simple('test','test')
current_backtest.perform_backtest_start_agnostic()
#current_backtest.run_strategy(shorten_backtest=True)


current_backtest.choosen_strat.df_opportunities_backlog.to_clipboard()

