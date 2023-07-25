# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:48:38 2021

@author: PC
"""
import logging
import numpy as np
import pandas as pd 
import datetime

import statsmodels.api as sm 
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.stattools import adfuller
from scipy import stats

from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore',ConvergenceWarning)

"""
https://sarit-maitra.medium.com/markov-regime-switching-non-linear-model-8ebfdf6eb755

https://github.com/saritmaitra/Markov_Model/blob/master/markov_regime_switching.py

"""




############  Let's define an input


#regime_A 

mu_A = 80000
sigma_A = 3000
sample_length_A = 200
std_A = mu_A + sigma_A * np.random.standard_normal(size=sample_length_A)
df_std_A = pd.DataFrame(std_A)
df_std_A['regime'] = 'A'
#regime_B 

mu_B = 30000
sigma_B = 5000
sample_length_B = 300
std_B = mu_B + sigma_B * np.random.standard_normal(size=sample_length_B)
df_std_B = pd.DataFrame(std_B)
df_std_B['regime'] = 'B'
#regime_C 

mu_C = 110000
sigma_C = 40000
sample_length_C = 400
std_C = mu_C + sigma_C * np.random.standard_normal(size=sample_length_C)
df_std_C = pd.DataFrame(std_C)
df_std_C['regime'] = 'C'
std_concat = pd.concat([df_std_A,df_std_B,df_std_C],axis=0,ignore_index=True)



step = '5m'

date_from_simulated = datetime.datetime(2021,3,15,0,0,0)

df_bt_15m = pd.read_excel('C:\Data_repo\crypto/05082021-18_08_fq_15mbt_btcusdt.xlsx')
std_concat.rename(columns={0:'Close'},inplace=True)


df_input_model = std_concat.copy()

df_input_model = df_bt_15m.copy()
"""
df_input_model.loc[:,'date_time'] = date_from_simulated
df_input_model['time_added'] = pd.to_timedelta(5*df_input_model.index,'m')
df_input_model.loc[:,'date_time'] = df_input_model.loc[:,'date_time']+df_input_model['time_added']
"""
"""
df_input_model['volume'] = np.random.randint(1000, 6000, df_input_model.shape[0])
df_input_model['Open'] = df_input_model['Close'] - np.random.randint(450, 550, df_input_model.shape[0])
df_input_model['High'] = df_input_model['Close'] + np.random.randint(450, 550, df_input_model.shape[0])
df_input_model['Low'] = df_input_model['Open'] - np.random.randint(450, 550, df_input_model.shape[0])
"""
adfuller(df_input_model['Close'].dropna())


##### lets calculate log returns

df_input_model.loc[:,'log_returns'] = np.log(df_input_model['Close'] / df_input_model.shift(periods=1, axis=0)['Close'])

df_input_model.loc[:,'log_returns'].hist(bins=30)

df_input_model.to_clipboard()

stats.probplot(df_input_model['Close'], plot=plt);

####### model fitting
df_input_model = df_input_model.fillna(df_input_model.mode().iloc[0])  

df_to_model = df_input_model.copy()

mod_k_regimes = 3
mod_order = 1
mod_switching_ar = True
mod_switching_trend = False
mod_switching_variance = True
column_to_model = df_bt_15m['log_returns'][-2000:]
#column_to_model =  df_input_model['log_returns'][-1000:]

mod_btc = sm.tsa.MarkovAutoregression(
     column_to_model
    ,k_regimes=mod_k_regimes
    ,order=mod_order
    , switching_ar=mod_switching_ar
    #,switching_exog=True
    #,switching_trend=mod_switching_trend
    , switching_variance=mod_switching_variance)

np.random.seed(123)

res_mod = mod_btc.fit(search_reps=1000)
res_mod.summary()
print("expected duration",res_mod.expected_durations)
fig, axes = plt.subplots(5,figsize=(10,7))

now = datetime.datetime.now()
ts = now.isoformat(sep='-')

fp_modeler = 'C:/PythonWorkspace/owomodo/owomodo_scrapping/financial_library/prodable_a_la_casa/Cistercian_bot/markovRegimeModeller/'
df_previous_model_result = pd.read_excel(fp_modeler+'Summary_model_result.xlsx')
add_new_row = df_previous_model_result.index.max() + 1
df_previous_model_result.loc[add_new_row,'ts'] = ts

df_previous_model_result.loc[add_new_row,'mu_A'] = mu_A
df_previous_model_result.loc[add_new_row,'mu_B'] = mu_B
df_previous_model_result.loc[add_new_row,'mu_C'] = mu_C
df_previous_model_result.loc[add_new_row,'sigma_A'] = sigma_A
df_previous_model_result.loc[add_new_row,'sigma_B'] = sigma_B
df_previous_model_result.loc[add_new_row,'sigma_C'] = sigma_C
df_previous_model_result.loc[add_new_row,'mod_k_regimes'] = mod_k_regimes
df_previous_model_result.loc[add_new_row,'mod_order'] = mod_order
df_previous_model_result.loc[add_new_row,'mod_switching_ar'] = mod_switching_ar
df_previous_model_result.loc[add_new_row,'mod_switching_trend'] = mod_switching_trend
df_previous_model_result.loc[add_new_row,'mod_switching_variance'] =mod_switching_variance
df_previous_model_result.loc[add_new_row,'aic'] =res_mod.aic
df_previous_model_result.loc[add_new_row,'bic'] =res_mod.bic

try : 
    df_previous_model_result.loc[add_new_row,'pvalue_const0']=res_mod.pvalues['const[0]']
    df_previous_model_result.loc[add_new_row,'pvalue_const1']=res_mod.pvalues['const[1]']
    df_previous_model_result.loc[add_new_row,'pvalue_const2']=res_mod.pvalues['const[2]']
    df_previous_model_result.loc[add_new_row,'pvalue_sigma20']=res_mod.pvalues['sigma2[0]']
    df_previous_model_result.loc[add_new_row,'pvalue_sigma21']=res_mod.pvalues['sigma2[1]']
    df_previous_model_result.loc[add_new_row,'pvalue_sigma22']=res_mod.pvalues['sigma2[2]']
except KeyError:
    pass
df_previous_model_result.reset_index()

df_previous_model_result.to_excel(fp_modeler+'Summary_model_result.xlsx',index=False)

ax = axes[0]
ax.plot(column_to_model,color='crimson')
ax.set(title='simulated BTC Close price')

ax = axes[1]
ax.plot(res_mod.smoothed_marginal_probabilities[0])
ax.set(title='smoothed probability of a low variance regime for Close')

ax = axes[2]
ax.plot(res_mod.smoothed_marginal_probabilities[1])
ax.set(title='smoothed probability of a high variance regime for Close')


ax = axes[3]
ax.plot(res_mod.smoothed_marginal_probabilities[2])
ax.set(title='smoothed probability of a high variance regime for Close')


try : 
    ax = axes[4]
    ax.plot(res_mod.smoothed_marginal_probabilities[3])
    ax.set(title='smoothed probability of a high variance regime for Close')
except KeyError:
    pass


fig.savefig(fp_modeler+str(ts).replace(':','_').replace('.','_')+'full_figure.png')
res_mod.summary()

#####################################################
# PPredict 1 
###################################################
pred_window = 100
try : 
    pred_1 = res_mod.predict()

    pred_1 = pd.DataFrame(pred_1).tail(pred_window)
    pred_1.rename(columns ={0: 'Predicted'}, inplace=True)

    com_1 = pd.concat([pred_1, df_input_model.loc[:,('log_returns','date_time')].tail(pred_window)], axis=1)

    com_1 = com_1.reset_index()

    com_1.plot(x='date_time')

    fig, axes = plt.subplots(3, figsize=(10,7))
    ax = axes[0]
    ax.plot(res_mod.smoothed_marginal_probabilities[0])
    #ax.fill_between(data['returns'].index, 0,  where=data['returns'].values, color='gray', alpha=0.3)
    ax.set(title='Smoothed probability of down regime for Nasdaq returns')
    ax = axes[1]
    ax.plot(res_mod.smoothed_marginal_probabilities[1])
    ax.set(title='Smoothed probability of no_change regime for Nasdaq returns')
    ax = axes[2]
    ax.plot(res_mod.smoothed_marginal_probabilities[2])
    ax.set(title='Smoothed probability of up-variance regime for Nasdaq returns')
    fig.tight_layout()
except NotImplementedError:
    logging.error('pred not implemented out of samples problem')

df_input_model['volumeGap'] = np.log(df_input_model['Volume'] / df_input_model['Volume'].shift())
df_input_model['dailyChange'] = (df_input_model['Close'] - df_input_model['Open']) / df_input_model['Open']
df_input_model['fractHigh'] = (df_input_model['High'] - df_input_model['Open']) / df_input_model['Open']
df_input_model['fractLow'] = (df_input_model['Open'] - df_input_model['Low']) / df_input_model['Open']
df_input_model['forecastVariable'] = df_input_model['Close'].shift(-1) - df_input_model['Close']
df_input_model.dropna(inplace=True)
df_input_model = df_input_model[~df_input_model.isin([np.nan, np.inf, -np.inf]).any(1)]
endog = df_input_model['Close'][-2500:]
exog = df_input_model [['volumeGap', 'dailyChange', 'fractHigh', 'fractLow']][-2500:]

mod_2 = sm.tsa.MarkovRegression(endog=endog, k_regimes=3, exog=exog)
res_2 = mod_2.fit(search_reps=50)
print(res_2.summary())
print(res_2.expected_durations)

res_2.smoothed_marginal_probabilities[0].plot(
    title='Probability of being in a low-variance regime', figsize=(12,5));


fig, axes = plt.subplots(3, figsize=(10,7))

ax = axes[0]
ax.plot(res_2.smoothed_marginal_probabilities[0])
ax.set(title='Smoothed probability of down regime')

ax = axes[1]
ax.plot(res_2.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of no_change regime')

ax = axes[2]
ax.plot(res_2.smoothed_marginal_probabilities[2])
ax.set(title='Smoothed probability of up regime')

plt.tight_layout()

print(res_2.expected_durations)


# Plot a simple histogram with binsize determined automatically
sns.distplot(res_2.resid, 20)
plt.title('Histogram of residuals')
plt.xlabel('Residuals')
plt.ylabel('Density')
plt.grid(True)
plt.show()

from statsmodels.compat import lzip
import statsmodels.stats.api as sms

name = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
results1 = sms.acorr_breusch_godfrey(res_2, 10)
print(lzip(name, results1))

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
JB, JBpv,skw,kurt = sm.stats.stattools.jarque_bera(res_2.resid)
print(lzip(name, results1))

pred_2 = res_2.predict()
pred_2 = pd.DataFrame(pred_2.tail(20))
pred_2.rename(columns ={0: 'Predicted'}, inplace=True)
combine = pd.concat([pred_2, df_input_model.loc[:,('log_returns','date_time')].tail(20)], axis=1)
combine = combine.reset_index()
combine.loc[:,('date_time','log_returns','Predicted')].plot(x='date_time',secondary_y='Predicted')


