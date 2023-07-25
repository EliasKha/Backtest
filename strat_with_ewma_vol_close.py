# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 18:05:52 2020

@author: rr5862



"""

# coding=utf-8

import ccxt
import pandas as pd
import datetime
import numpy as np
import seaborn as sn

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt
sn.set_style('darkgrid')
pd.plotting.register_matplotlib_converters()
#%matplotlib inline

#from sklearn import linear_model




import library_instruments as lib
import graph_nd_stats_research as libresearch



hitbtc   = ccxt.hitbtc({'verbose': True})
#bitmex   = ccxt.bitmex()
huobipro = ccxt.huobipro()
kraken = ccxt.kraken()
deribit = ccxt.deribit()




#hitbtc_markets = hitbtc.load_markets()

#print(hitbtc.id, hitbtc_markets)
"""
print(bitmex.id, bitmex.load_markets())
print(huobipro.id, huobipro.load_markets())
"""
#aa = bitmex.fetch_ticker('BTC/USD')
#bitmex.fetchOHLCV('.BXBT','1h')
#df = pd.DataFrame(aa)
#




class another_time_series():
    def __init__(self):
        self.state = 'Inactive'
        self.min_window_rolling = 5
        
    def load_binance_ts(self,market_requested='BTC/USDT'
                        ,frequency_requested = '1h'
                        ,from_datetime ='15/02/2020 07:00:00' 
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
        
        
        self.market = 'BTC/USDT'
        self.exchange_name = 'binance'
        self.exchange = ccxt.binance({
            'rateLimit': 10000,
            'enableRateLimit': True,
            # 'verbose': True,
        })
        
        self.markets = self.exchange.load_markets()
        #self.from_timestamp = self.exchange.parse8601(from_datetime)
        self.dt_from_datetime = datetime.datetime.strptime(from_datetime,'%d/%m/%Y %H:%M:%S')

        self.from_timestamp = int(self.dt_from_datetime.timestamp()*1000)

        candles = self.exchange.fetch_ohlcv(market_requested, frequency_requested, self.from_timestamp)
        self.df_candles_binance = pd.DataFrame(candles, columns=['Timestamp','Open','High','Low','Close', 'Volume'])
        self.df_candles_binance['date_time'] = pd.to_datetime(self.df_candles_binance['Timestamp'],unit='ms')
        self.df_under_study = self.df_candles_binance.copy()
        
    
    def zoom_in_time_axis(self,start_time='2020-02-13 00:00:00',end_time='2020-02-27 00:00:00'):
        df = self.df_candles_binance.copy()
        self.df_under_study = df[(df['date_time'] >= start_time) & (df['date_time'] <= end_time)].copy()
        
        
     
class performance_indicators():
    
    def __init__(self):
        self.perf_status='initiated'
    
    def calculate_performance_indicators():
        pass
    
        
        
        
        

class deribit_algo_strat(another_time_series,libresearch.research_indicators,performance_indicators):
    def __init__(self):
        super().__init__()
        self.state = 'OK'
        self.min_window_rolling = 5
        self.path_to_save = 'C:/PythonWorkspace/sandbox_outputs/strat_bot_cryptos/'


    def reglin_rolling_stats_model_time(self,df):
        
        endog = df.loc[:,'Close']
        exog = sm.add_constant(df['Timestamp'])
        rols = RollingOLS(endog, exog, window=15)
        rres = rols.fit()
        df.loc[:,'const'] = rres.params.iloc[:,0]
        df.loc[:,'ax_timestamp'] = rres.params.iloc[:,1]
        df.loc[:,'approx'] = df.loc[:,'const']+df.loc[:,'ax_timestamp']*df.loc[:,'Timestamp']
        df.to_clipboard()
        params = rres.params

        
        

    

    def calculate_relative_returns(self,df, retcolums):
        """calculates the normalized the returns"""
        colname = retcolums + '_shifted'
        df[colname] = df.shift(periods=1, axis=0)[retcolums]
        df['returns'] = df[retcolums] - df[colname]        
        df['returns_relative'] = df['returns'] / df[colname]    
        df['log_returns'] = np.log(df[retcolums] / df[colname] )        
        df['returns_relative'].fillna(value=0, inplace=True)        
        df['log_returns'].fillna(value=0, inplace=True)
        # lets remove the -inf , replace by 0 
        df['log_returns'].replace('=-inf',0)
        return df
    
    def calculate_rolling_metrics(self,df,to_be_calculated_col):
        self.calculate_relative_returns(df,to_be_calculated_col)

        roll_mean_suffix = '_roll_mean'
        roll_ewma_st_dev_suffix = '_ewma_std_dev'
        
        current_returns_col_names = 'log_returns'
        
        # on prices 
        mean_roll_name = to_be_calculated_col+roll_mean_suffix
        
        ewma_std_name = to_be_calculated_col+roll_ewma_st_dev_suffix
        df.loc[:,mean_roll_name] = df[to_be_calculated_col].rolling(window=25, min_periods=self.min_window_rolling).mean()
        df.loc[:,ewma_std_name] = df[to_be_calculated_col].ewm(span=7).std()
        # on returns
        mean_roll_returns = current_returns_col_names+roll_mean_suffix
        ewma_std_dev_returns = current_returns_col_names+roll_ewma_st_dev_suffix
        try:
            df.loc[:,mean_roll_returns] = df[current_returns_col_names].rolling(window=25, min_periods=self.min_window_rolling).mean()
            df.loc[:,ewma_std_dev_returns] = df[current_returns_col_names].ewm(span=7).std()
        except KeyError:
            print('no log_returns?')
        
        
        
        return df
    
    def get_calculate_close_binance(self
                                    ,frequency_requested = '1h'
                                    ,from_datetime ='15/02/2020 07:00:00' 
                                   ):

                        
        
        self.load_binance_ts(market_requested='BTC/USDT',frequency_requested = frequency_requested
                             ,from_datetime=from_datetime)
        
    def calculate_performances(self,df, capital_in_btc=1):
        """
        
        

        Parameters
        ----------
        df : TYPE, optional
            DESCRIPTION. The default is self.df_merged.
        capital_in_btc : ## NOT USED YET ## TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        cum_sum_performance : TYPE
            DESCRIPTION.

        """
        
        buy_price =0 
        cum_sum_performance=0
        try: 
             
            for i in df.index:
                if not(pd.isnull(df.loc[i,'buy_price'])):
                    # buy price cell is not empty 
                    buy_price = df.loc[i,'buy_price']
                elif not(pd.isnull(df.loc[i,'sell_price'])):
                    sell_price = df.loc[i,'sell_price']
                    df.loc[i,'net_preformance'] = sell_price - buy_price
        except KeyError:
            pass
            #print('error on : ',df.loc[i,'sell_price'])
                    
        try:        
            cum_sum_performance += df.loc[:,'net_preformance'].sum()
        except (UnboundLocalError,KeyError):
            pass
        return cum_sum_performance
        
        
    
    def slope_btc_derivative_plus_vol_plus_fwd_funnel(self,freq='1m'
                                                  ,from_timestamp='15/02/2020 07:00:00'
                                                  , merged_tolerance_minutes = 2
                                                  ,threshold_up = 0.00004
                                                  ,threshold_down=-0.00004
                                                  ,init_amount_usd = 10000.0
                                                  ):
        
        """
        final strat on all mentionned factors
        """
        
        self.get_calculate_close_binance(freq,from_timestamp)
        self.calculate_relative_returns(self.df_under_study,'Close')
        self.reglin_rolling_stats_model_time(self.df_under_study)
        
        # from timstamp to be inverted  , correct format : 2020-02-07 17:04:30.0
        from_timestamp_dt = datetime.datetime.strptime(from_timestamp,'%d/%m/%Y %H:%M:%S')
        
        
        self.df_shlomo = libresearch.test_cheap_call_week1_expensive_week3(from_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S'),100000)
        
        #filters on shlomo to get the appropriate vol 
        
        self.df_shlomo = self.df_shlomo[(self.df_shlomo.loc[:,'option_type'] == 'call')
            & (self.df_shlomo.loc[:, 'time_to_maturity'] < datetime.timedelta(days=6))
            & (self.df_shlomo.loc[:, 'option_moneyness'].between(0.78,0.83,inclusive=True))]
        
        
        
        self.df_under_study['capture_ts'] = pd.to_datetime(self.df_under_study['date_time'], format='%Y-%m-%d %H:%M:%S')
        self.df_shlomo['capture_ts_shlomo'] = pd.to_datetime(self.df_shlomo['capture_ts'], format='%Y-%m-%d %H:%M:%S')
        self.df_shlomo['capture_ts_shlomo'] = pd.to_datetime(self.df_shlomo['capture_ts'], format='%Y-%m-%d %H:%M:%S')


        self.df_under_study.index = self.df_under_study['capture_ts']
        self.df_merged = pd.merge_asof(left=self.df_under_study,right=self.df_shlomo, left_index=True, right_index=True, direction='nearest'
                                       ,tolerance=pd.Timedelta(minutes=merged_tolerance_minutes))
        
        
        # il faut rajouter la derivée de ax par rapport au temps : 
        self.df_merged.loc[:,'ax_timestamp_-1'] =  self.df_merged.loc[:,'ax_timestamp'].shift(periods=1, axis=0)
        self.df_merged.loc[:,'dt'] = self.df_merged.loc[:,'date_time'].diff().dt.seconds.values
        self.df_merged.loc[:,'dax/dt'] = self.df_merged.loc[:,'ax_timestamp'] - self.df_merged.loc[:,'ax_timestamp_-1']
        self.df_merged.loc[:,'dax/dt'] = self.df_merged.loc[:,'ax_timestamp'] - self.df_merged.loc[:,'ax_timestamp_-1']
        self.df_merged.loc[:,'dax/dt'] = self.df_merged.loc[:,'dax/dt'] / self.df_merged.loc[:,'dt']
        self.df_merged.loc[:,'dax/dt'] = 10000.0*self.df_merged.loc[:,'dax/dt']
        self.calculate_rolling_metrics(self.df_merged,'log_returns')
        self.calculate_rolling_metrics(self.df_merged,'Close')
        
        #lets apply a strategy
        
        # for 1 m frequency, threshold = +- 0.001 was working 
        # for 5m , I'd say -+ 9e-5 
        
        threshold_buy = threshold_up
        threshold_sell = threshold_down
        current_state = 'cash'
        init_amount_usd = 10000.0
        
        
        states = ['cash','btc']
        
        
        
        for i in self.df_merged.index:
            try:
                # lets first consider safty locks : 
                if (self.df_merged.loc[i,'Close_ewma_std_dev']>200.0)& (current_state=='btc')& (self.df_merged.loc[i,'log']=='buy due to vol'):
                    #stop loss in vol 
                    self.df_merged.loc[i,'sell_price'] = self.df_merged.loc[i,'Close']
                    self.df_merged.loc[i,'capital_state_tether'] = 1
                    self.df_merged.loc[i,'capital_state_btc'] = 0
                    self.df_merged.loc[i,'log']='stop_loss_vol'
                    current_state = 'cash' 
                    continue                                     
                    
                if (self.df_merged.loc[i,'Close']<self.df_merged.loc[i,'buy_price']*0.9) & (current_state=='btc'):
                    #stop loss 10% in price 
                    self.df_merged.loc[i,'sell_price'] = self.df_merged.loc[i,'Close']
                    self.df_merged.loc[i,'capital_state_tether'] = 1
                    self.df_merged.loc[i,'capital_state_btc'] = 0
                    self.df_merged.loc[i,'log']='stop_loss_price'
                    current_state = 'cash' 
                    continue
            except KeyError:
                self.df_merged.loc[i,'log'] ='no buy price'
                #self.df_merged.to_excel(self.path_to_save+'aa.xlsx')
            
            try:
                if (self.df_merged.loc[i,'dax/dt'] > 0.0+threshold_buy)&(self.df_merged.loc[i,'Close_ewma_std_dev'] < 20.0)&(current_state=='cash'):
                    # buy on derivative 
                    # /!\! ce n'est pas le close mais le prochain open !! 
                    self.df_merged.loc[i,'buy_price'] = self.df_merged.loc[i,'Close']
                    current_state = 'btc'
                    self.df_merged.loc[i,'capital_state_btc'] = 1
                    self.df_merged.loc[i,'capital_state_tether'] = 0
                    self.df_merged.loc[i,'log'] ='buy due derivative threshold'
                    continue
                
                
                elif (self.df_merged.loc[i,'Close_ewma_std_dev'] < 20.0)&(current_state=='cash'):
                    # buy due to low vol 
                    self.df_merged.loc[i,'buy_price'] = self.df_merged.loc[i,'Close']
                    current_state = 'btc'
                    self.df_merged.loc[i,'capital_state_btc'] = 1
                    self.df_merged.loc[i,'capital_state_tether'] = 0
                    self.df_merged.loc[i,'log'] ='buy due to vol'
                    continue
                
                                                  
                    
                    
                elif (self.df_merged.loc[i,'dax/dt'] < 0.0+threshold_sell)&(current_state=='btc'):
                    # sell due to low derivative of slope 
                    self.df_merged.loc[i,'sell_price'] = self.df_merged.loc[i,'Close']
                    self.df_merged.loc[i,'capital_state_tether'] = 1
                    self.df_merged.loc[i,'capital_state_btc'] = 0
                    current_state = 'cash'
                    self.df_merged.loc[i,'log'] ='sell due to low derivative of slope'
                
                elif(self.df_merged.loc[i,'Close_ewma_std_dev'] > 90.0)&(current_state=='btc'):
                    # sell due to low ewma vol  
                    self.df_merged.loc[i,'sell_price'] = self.df_merged.loc[i,'Close']
                    self.df_merged.loc[i,'capital_state_tether'] = 1
                    self.df_merged.loc[i,'capital_state_btc'] = 0
                    current_state = 'cash'
                    self.df_merged.loc[i,'log'] ='sell due to low volatily trigger' 
            except KeyError:
                self.df_merged.loc[i,'log'] ='error in missing sell or buy price '
                                                                    
                
        return self.calculate_performances(self.df_merged)
        
        
        # puis grapher dax/dt , close et implied_vol 

        
        
    def plot_binance_df_ex_post_metrics(self,display_graph=False):
        
        """
        takes previous 2 functions and plot them
        """
        
        self.calculate_rolling_metrics(self.df_under_study,'Close')
        self.df_under_study.loc[:,'log_returns'].hist(bins=30);
        self.df_under_study.plot(x='date_time'
                                     ,y=['log_returns','log_returns_roll_mean','log_returns_ewma_std_dev']
                                     ,secondary_y='log_returns_ewma_std_dev')
        self.df_under_study.plot(x='date_time'
                                     ,y=['Close','Close_roll_mean','Close_ewma_std_dev']
                                     ,secondary_y='Close_ewma_std_dev')
        
        # ex post measures _ NOT TO BE TAKEN IN PROD !!! 
        """
        self.df_under_study.loc[:,'log_returns_std_dev'] = self.df_under_study.loc[:,'log_returns'].std()
        self.df_under_study.loc[:,'close_std_dev'] = self.df_under_study.loc[:,'Close'].std()
        # VaR @ 95% : 2 sigmas 
        self.df_under_study.loc[:,'log_returns_max'] = self.df_under_study.loc[:,'log_returns_roll_mean']+2*self.df_under_study.loc[:,'log_returns_std_dev']
        self.df_under_study.loc[:,'log_returns_min'] = self.df_under_study.loc[:,'log_returns_roll_mean']-2*self.df_under_study.loc[:,'log_returns_std_dev']
        self.df_under_study.plot(x='date_time',y=['log_returns','log_returns_min','log_returns_max','log_returns_ewma_std_dev'],secondary_y='log_returns_ewma_std_dev');
        self.df_under_study.loc[:,'close_returns_max'] = self.df_under_study.loc[:,'Close_roll_mean']+2*self.df_under_study.loc[:,'close_std_dev']
        self.df_under_study.loc[:,'close_returns_min'] = self.df_under_study.loc[:,'Close_roll_mean']-2*self.df_under_study.loc[:,'close_std_dev']
        self.df_under_study.plot(x='date_time'
                             ,y=['Close','close_returns_min','close_returns_max','log_returns_ewma_std_dev']
                             ,secondary_y='log_returns_ewma_std_dev');
        """
        
class research_histo_deribit(deribit_algo_strat):
    def __init__(self):
        super().__init__()
        self.status = 'not_started'
    def get_nd_merge_deribit_binance_mktdata(self,tol='5 minutes',freq='1h',inquiry_start_date='15/02/2020 07:00:00',inquiry_end_date='27/02/2020 17:00:00'):
        tolerance_to_merge = tol
        self.get_calculate_close_binance(freq,inquiry_start_date)
        
        self.zoom_in_time_axis(inquiry_start_date, inquiry_end_date)
        self.calculate_relative_returns(self.df_under_study,'Close')
        self.plot_binance_df_ex_post_metrics()
        
        #test2
        self.df_under_study['capture_ts'] = pd.to_datetime(self.df_under_study['date_time'], format='%Y-%m-%d %H:%M:%S')
        self.df_under_study.index = self.df_under_study['capture_ts']
        
        sql_trunc_start = inquiry_start_date[:10]
        sql_trunc_stop = inquiry_end_date[:10]
        
        
        self.df_shlomo_perpetual = libresearch.query_a_given_future(instrument_name='BTC-PERPETUAL',start_date = sql_trunc_start,end_date=sql_trunc_stop,with_graph=False)
        self.df_shlomo_perpetual['capture_ts_shlomo'] = pd.to_datetime(self.df_shlomo_perpetual['capture_ts'], format='%Y-%m-%d %H:%M:%S')
        self.df_shlomo_perpetual.index = self.df_shlomo_perpetual['capture_ts_shlomo']
        self.df_merged = pd.merge_asof(left=self.df_under_study,right=self.df_shlomo_perpetual, left_index=True, right_index=True, direction='nearest',tolerance=pd.Timedelta('15 minute'))
        
        
        
        
        self.df_OTM_call = libresearch.plot_a_given_instrument(
            instrument_name='BTC-26JUN20-12000-C'
            ,start_date = sql_trunc_start
            ,end_date=sql_trunc_stop 
            )
        self.df_OTM_call['capture_ts_shlomo'] = pd.to_datetime(self.df_OTM_call['capture_ts'], format='%Y-%m-%d %H:%M:%S')
        
        self.df_OTM_call.index = self.df_OTM_call['capture_ts_shlomo']
        self.df_merged = pd.merge_asof(left=self.df_merged,right=self.df_OTM_call, left_index=True, right_index=True, direction='nearest',
                           tolerance=pd.Timedelta(tol))
        
        
        
        self.df_merged.to_clipboard()
        
        

        """df_result.index = df_result['timestamp']
        df_iv['capture_ts'] = pd.to_datetime(df_iv['capture_ts'], format='%Y-%m-%d %H:%M:%S')
        df_iv.index = df_iv['capture_ts']
        aa = pd.merge_asof(left=self.df_under_study, right=self.df_shlomo_perpetual, left_index=True, right_index=True, direction='nearest',
                           tolerance=tolerance_to_merge)
        aa.to_clipboard()
        """




     

# sur toute la base : 
    # faire N scénarios à la minute 
    # calculer les perfs sur la strategy 
    # sortir un df de résultat complet avec et sans le scénario covid




        

"""
to KEEP , deribits tries !!!! 
deribit.fetchTickers(symbols='BTC-17APR20-6750-P')
orderbook = deribit.fetch_order_book (deribit.symbols[0])
bid = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
ask = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
spread = (ask - bid) if (bid and ask) else None
df = pd.DataFrame(deribit.fetchOrderBook('BTC-17APR20-6750-P'))
"""

        
        
# lets get 1month BTCUSDT return , plot it 


#BTC-PERPETUAL
if __name__ == '__main__':
    """
    obj_test = deribit_algo_strat()
    obj_test.get_calculate_close_binance('1h','15/02/2020 07:00:00')
    
    obj_test.zoom_in_time_axis('15/02/2020 07:00:00','27/02/2020 15:00:00')
    obj_test.calculate_relative_returns(obj_test.df_under_study,'Close')
    #test
    

    #obj_test.plot_binance_df_ex_post_metrics()
    #obj_test.df_candles_binance.to_clipboard()
    merj = research_histo_deribit()
    merj.get_nd_merge_deribit_binance_mktdata(tol='5 minutes',freq='1h',inquiry_start_date='15/02/2020 07:00:00',inquiry_end_date='27/02/2020 17:00:00')
    
    merj.df_merged.plot(x='date_time',y=['Close','log_returns','log_returns_ewma_std_dev']
                             ,secondary_y=['log_returns','log_returns_ewma_std_dev'])
    
    merj.df_merged.plot(x='date_time',y=['Close','log_returns','log_returns_ewma_std_dev','implied_vol']
                             ,secondary_y=['log_returns','log_returns_ewma_std_dev','implied_vol'])
    
    """
    
    obj_test = deribit_algo_strat()
    
    freq = '5m'
    tol_for_merge=31
    """
    
        def slope_btc_derivative_plus_vol_plus_fwd_funnel(self,freq='1m'
                                                  ,from_timestamp='15/02/2020 07:00:00'
                                                  , merged_tolerance = '5 minute'
                                                  ,threshold_up = 0.001
                                                  ,threshold_down=0.001
                                                  ,init_amount_usd = 10000.0
                                                  ):
    """
            
        
    """
    res = obj_test.slope_btc_derivative_plus_vol_plus_fwd_funnel(freq
                                                                 ,'03/04/2020 08:00:00'
                                                                 ,tol_for_merge)
    path_to_save = 'C:/PythonWorkspace/sandbox_outputs/strat_bot_cryptos/'
    
    
    obj_test.calculate_rolling_metrics(obj_test.df_merged,'log_returns')
    obj_test.calculate_rolling_metrics(obj_test.df_merged,'Close')
    df_tocopypaste = obj_test.df_merged.copy()
    df_tocopypaste.replace(to_replace='.',value=',',inplace=True)
    df_tocopypaste.to_excel()
    df_tocopypaste.to_clipboard()
    # ça trace 3 graphes 
    df_tocopypaste.plot(x='Timestamp',y=['Close','dax/dt'],secondary_y='dax/dt')
    df_tocopypaste.plot(x='Timestamp',y=['implied_vol','dax/dt'],secondary_y='dax/dt')
    df_tocopypaste.plot(x='Timestamp',y=['Close','implied_vol'],secondary_y='implied_vol')
    
    print('la somme des benefs =', res)
    
    """
    #lets create a date array for testing multiple dates on the strategy
    
    nb_test_points = 45
    ts_start_period = datetime.datetime(2020, 3, 19, 10, 0)
    path_to_save = 'C:/PythonWorkspace/sandbox_outputs/strat_bot_cryptos/'
    date_array = []
    res_array=[]
    res=0.0
    dict_summary_results = {ts_start_period:res}
    for i in range(nb_test_points):
        date_to_add = ts_start_period+datetime.timedelta(hours=40*i)
        date_array.append(date_to_add)
        #date_array.append(date_to_add.strftime('%d/%m/%Y %H:%M:%S'))
        
    # then lets test the strat for every test set and print the result
    for i in date_array:
        res = obj_test.slope_btc_derivative_plus_vol_plus_fwd_funnel('5m',i.strftime('%d/%m/%Y %H:%M:%S'),5)
        obj_test.calculate_rolling_metrics(obj_test.df_merged,'log_returns')
        obj_test.calculate_rolling_metrics(obj_test.df_merged,'Close')
        res_array.append(res)
        print(path_to_save+'scenar_'+str(i.strftime('%d-%m-%Y_%H-%M-%S'))+'_'+'.xlsx')
        obj_test.df_merged.to_excel(path_to_save+'scenar_'+'res_'+(str(res)[:5])+'_tol_freq_'+str(freq)+'_'+str(tol_for_merge)+'_'+str(i.strftime('%d-%m-%Y_%H-%M-%S'))+'_'+'.xlsx')
        print('la somme des benefs pour : ',str(i),'est : ', res)
        
    
    
        
        
    # rajouter le wait !!! 
    
    # chopper un shlomo , 
    # donner l'IV d'un call à 0.85 de moneyness 
    # comparer sur l'histo comment ça prédit les chutes '