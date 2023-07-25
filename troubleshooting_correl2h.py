# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 23:03:39 2021

@author: PC
"""

import pandas as pd
import seaborn as sns
import numpy as np 
import perform_strat_upon_backtest as perf_strat


#test de la correl 
oj =  perf_strat.backtest_crypto_strategy_simple('test','test')
oj.load_backtest_scenario('15mn')

strat = perf_strat.another_algo_strategy('2h_correlations',oj.df_backtest)

strat.df_bt_15m = oj.df_backtest.copy()
#strat.calculate_2h_correl()
#strat.initiate_2h_correl()

strat.use_correlogram_and_give_predictions()

print(strat.df_use_time_period)



"""
#test criterion 

strat.df_corr[(strat.df_corr>0.5)&(strat.df_corr!=1.0)]
val = np.nan
is_null = pd.isnull(val)
df2 = ~pd.isnull(strat.df_corr[(strat.df_corr>0.5)&(strat.df_corr!=1.0)])
df_result_corr_criterion = strat.df_corr.where((strat.df_corr>0.5) | (strat.df_corr<-0.5))
bx = sns.heatmap(strat.df_corr, cmap="Blues", linewidth=0.3, annot=True, fmt=".2f", cbar_kws={"shrink": .8},palette=sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True))

rows_to_look_at = strat.df_corr.where((strat.df_corr>0.5)&(strat.df_corr!=1.0)).dropna(axis=0,how='all').index

# this confirms that we are in the good direction : 
if (len(rows_to_look_at)==2) and (strat.df_corr.loc[rows_to_look_at[0], rows_to_look_at[1]] > strat.pos_corr_criterion):
    print('yippekaï pauvre con')
    

#this puts in form the desired pair of result, still duplicated : 

df_res = pd.DataFrame(columns=['hour_pair1','hour_pair2','corr'])
for i in df_result_corr_criterion : 
    for j in df_result_corr_criterion.columns:
        print(df_result_corr_criterion.loc[i,j])
        if df_result_corr_criterion.loc[i,j]>0.5:
            df_res.loc[i+'_'+j,'hour_pair1'] = i
            df_res.loc[i+'_'+j,'hour_pair2'] = j
            df_res.loc[i+'_'+j,'corr'] = df_result_corr_criterion.loc[i,j]
        if df_result_corr_criterion.loc[i,j]<-0.5:
            df_res.loc[i+'_'+j,'hour_pair1'] = i
            df_res.loc[i+'_'+j,'hour_pair2'] = j
            df_res.loc[i+'_'+j,'corr'] = df_result_corr_criterion.loc[i,j]
            

df_res = df_res[df_res['corr']!=1]
#let's remove the non complian (late correl) and keep the buy early sell later 
df_res[df_res['hour_pair1']<df_res['hour_pair2']]


df_res = df_res.reset_index()
#del(df_res['index'])
print(df_res)
"""
