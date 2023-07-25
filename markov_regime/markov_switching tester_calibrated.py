# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:48:38 2021

@author: PC
"""

import numpy as np
import pandas as pd 
import datetime

import statsmodels.api as sm 
import matplotlib.pyplot as plt
import warnings



from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore',ConvergenceWarning)


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
sigma_C = 20000
sample_length_C = 400
std_C = mu_C + sigma_C * np.random.standard_normal(size=sample_length_C)
df_std_C = pd.DataFrame(std_C)
df_std_C['regime'] = 'C'
std_concat = pd.concat([df_std_A,df_std_B,df_std_C],axis=0,ignore_index=True)



step = '5m'

date_from_simulated = datetime.datetime(2021,3,15,0,0,0)

std_concat.rename(columns={0:'Close'},inplace=True)
df_input_model = std_concat.copy()
df_input_model.loc[:,'date_time'] = date_from_simulated
df_input_model['time_added'] = pd.to_timedelta(5*df_input_model.index,'m')
df_input_model.loc[:,'date_time'] = df_input_model.loc[:,'date_time']+df_input_model['time_added']



##### lets calculate log returns

df_input_model.loc[:,'log_returns'] = np.log(df_input_model['Close'] / df_input_model.shift(periods=1, axis=0)['Close'])

df_input_model.loc[:,'log_returns'].hist(bins=30)

df_input_model.to_clipboard()

####### model fitting
df_input_model = df_input_model.fillna(df_input_model.mode().iloc[0])  

df_to_model = df_input_model.copy()

mod_k_regimes = 3
mod_order = 1
mod_switching_ar = True
mod_switching_trend = False
mod_switching_variance = True


mod_btc = sm.tsa.MarkovAutoregression(
    #df_bt_15m['Close']
    df_input_model['Close']
    ,k_regimes=mod_k_regimes
    ,order=mod_order
    , switching_ar=mod_switching_ar
    #,switching_exog=True
    #,switching_trend=mod_switching_trend
    , switching_variance=mod_switching_variance)

res_mod = mod_btc.fit()
res_mod.summary()
fig, axes = plt.subplots(5,figsize=(10,7))

now = datetime.datetime.now()
ts = now.isoformat(sep='-')

fp_modeler = 'YOUR_OWN_PATH/markovRegimeModeller/'
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
ax.plot(df_to_model.loc[:,'Close'],color='crimson')
ax.set(title='simulated Exog Close price')

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


