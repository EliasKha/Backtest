# -*- coding: utf-8 -*-
"""
# Created by PC at 04/08/2021

@author: Simon 

Let's fuck off  salim-Henri 

"""
import ccxt
import datetime
#import market_loader.market_loader_ccxt as market_loader
import logging
import pandas as pd
import numpy as np
import strat_with_ewma_vol_close as strat
import datetime
import parameters as params

import matplotlib.pyplot as plt
import seaborn as sns

#import xlsxwriter

logging.basicConfig(level=logging.INFO)


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
file_handler = RotatingFileHandler('Backtesting.log', 'a', 1000000, 1)
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
logger.info('premier test de logguer backlog')
logger.warning('Testing %s', 'foo')




class another_set_of_metrics():
    def __init__(self, name_metrics, df_under_study):
        self.name = name_metrics
        self.df_under_study = df_under_study

    def measure_rolling_ewma_std(self):
        """
        this method makes sure that all measures are ready in the back test and presents trigger range
        :param col_to_analysed:
        :return:
        """
        self.df_under_study = self.df_backtest
        self.metric1 = {'metric_name' :'log_returns_ewma_std_dev'}
        #result_name = 'res_' + col_to_analysed
        self.metric1['minimum'] = self.df_under_study.loc[:, 'log_returns_ewma_std_dev'].min()
        self.metric1['maximum'] = self.df_under_study.loc[:, 'log_returns_ewma_std_dev'].max()
        logger.info('min et max de ewm_std calculated ')


class another_algo_strategy():
    def __init__(self,strategy_name,dfstrat_under_study):
        self.name = strategy_name
        self.dfstrat_understudy = dfstrat_under_study
        #self.current_datetime = self.dfstrat_under_study['date_time'].tail(1).iloc[0]
        self.current_datetime = '0'
        self.tail_limit = 'NO'
        self.pos_corr_criterion = 0.4
        self.neg_corr_criterion = -0.4
        self.df = 'rt'
        self.df_previous_opportunities = 'empty'
        self.corr_output_path = '/Cistercian_bot/backtest_ressource/Correlograms/'
        self.correlogram_paths_full_histo = {'Monday':self.corr_output_path+'corr__Monday_full_correl.xlsx'
                                             ,'Tuesday':self.corr_output_path+'corr__Tuesday_full_correl.xlsx'
                                             ,'Wednesday':self.corr_output_path+'corr__Wednesday_full_correl.xlsx'
                                             ,'Thursday':self.corr_output_path+'corr__Thursday_full_correl.xlsx'
                                             ,'Friday':self.corr_output_path+'corr__Friday_full_correl.xlsx'
                                             ,'Saturday':self.corr_output_path+'corr__Saturday_full_correl.xlsx'
                                             ,'Sunday':self.corr_output_path+'corr__Sunday_full_correl.xlsx'}

        self.strategy_stop_loss = 0.15
    
    def calculate_2h_correl(self,choosen_day_of_week='Tuesday'):
        
        """
        this is new : we only want to update the correl df for a given day


        :return


        """

        self.Daystudied = choosen_day_of_week
        # lets take the backtesting df and create HH MM columns , and filter per 2h sets 
        self.df_bt_15m['day_of_week'] = self.df_bt_15m['date_time'].dt.day_name()
        self.df_bt_15m['HH-MM'] = self.df_bt_15m['date_time'].dt.strftime('%HH-%MM')
        self.df_bt_15m['date'] = self.df_bt_15m['date_time'].dt.strftime('%d-%m')
        self.df = self.df_bt_15m[self.df_bt_15m.loc[:, 'HH-MM'].isin(['06H-00M'
                                                          , '08H-00M'
                                                          , '10H-00M'
                                                          , '12H-00M'
                                                          , '14H-00M', '16H-00M', '18H-00M', '20H-00M'
                                                          , '22H-00M', '00H-00M', '02H-00M', '04H-00M'

                                                       ])].loc[:,
             ('Close', 'Volume', 'date_time', 'day_of_week', 'HH-MM', 'date')].copy()
        
        # lets calculate the rolling metrics                                                        
        self.pricing_tool_box = strat.deribit_algo_strat()
        self.df_metrics = self.pricing_tool_box.calculate_rolling_metrics(self.df, 'Close')
        if self.tail_limit=='NO':
            pass
        else : 
            self.df_metrics = self.df_metrics.tail(self.tail_limit).copy()


        #let's take the current day as day of study
        try:
            self.df_filtered_selected_days = self.df_metrics[self.df_metrics.loc[:, 'day_of_week'] == self.Daystudied].loc[:,
                     ('log_returns', 'HH-MM', 'date')].copy()
        except IndexError:
            pass
        try:
            self.new_df = self.df_filtered_selected_days[self.df_filtered_selected_days['date'] == self.df_filtered_selected_days['date'].unique()[0]].copy()
            self.new_df = self.new_df.rename({'log_returns': 'log_returns' + str(self.df_filtered_selected_days['date'].unique()[0])}, axis=1)
        except IndexError :
            return('no_modifs')
            pass
        del (self.new_df['date'])
        for i in self.df_filtered_selected_days['date'].unique()[1:]:
            print(i)
            self.new_df = pd.merge(left=self.new_df,
                              right=self.df_filtered_selected_days[self.df_filtered_selected_days['date'] == i],
                              how="outer",
                              on='HH-MM',
                              left_index=False,
                              right_index=False,
                              sort=False,
                              suffixes=("_x", "_y"),
                              copy=False,
                              indicator=False,
                              validate=None)
            self.new_df = self.new_df.rename({'log_returns': 'log_returns' + str(i)}, axis=1)
            del (self.new_df['date'])

        self.new_df.set_index('HH-MM', inplace=True)
        self.res = self.new_df.transpose(copy=True)
        self.res = self.res.loc[:, ~self.res.columns.duplicated()]
        self.df_corr = self.res.corr()
        # i'm really not sure on how the corr() is working ! 2 dates : -1 et 1 /
        figsize = (300, 225)

    def initiate_2h_correl(self,nb_day_choosen='All'):
        self.strategy_name = '2H_correlation_dedicace_jim_simmons'
        self.df_bt_15m = self.dfstrat_understudy.copy()

        if nb_day_choosen == 'All':
            days_to_update = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        else :
            days_to_update = 'Tuesday'

        for i in days_to_update:
            print(i)
            self.tail_limit = 'NO'
            self.calculate_2h_correl(i)

            self.df_corr.to_excel(self.corr_output_path+"corr__"+str(i)+"Tail-"+str(self.tail_limit)+"_full_correl.xlsx")

            self.tail_limit = 4000
            self.calculate_2h_correl(i)

            self.df_corr.to_excel(
                self.corr_output_path + "corr__" + i + "Tail-" + str(self.tail_limit) + "_full_correl.xlsx")

    def load_2h_correlogram(self,choosen_day='Friday'):
        self.df_loaded_correlogram = pd.read_excel(self.correlogram_paths_full_histo[choosen_day],index_col=0)
    def load_2h_correlogram_with_tail(self,file_path='/home/simon/PythonWorkspace/Cistercian_bot/backtest_ressource/Correlograms/corr__FridayTail-4000_full_correl.xlsx'):
        self.df_loaded_correlogram = pd.read_excel(file_path,index_col=0)

    def load_df_metrics(self):
        # this is the same routine as calculate 2h correl without the correl calculation (only df metrics and rolling metrics for returns )
        # lets calculate the rolling metrics
        self.now = datetime.datetime.now()

        self.pricing_tool_box = strat.deribit_algo_strat()
        self.lookback_time = self.now - datetime.timedelta(days=1.2)
        # la partition de df under study en tranches de 2h n'est pas faite <--- Done
        self.pricing_tool_box.get_calculate_close_binance('15m',from_datetime =datetime.datetime.strftime(self.lookback_time,'%d/%m/%Y %H:%M:%S'))
        self.pricing_tool_box.df_under_study['day_of_week'] = self.pricing_tool_box.df_under_study['date_time'].dt.day_name()
        self.pricing_tool_box.df_under_study['HH-MM'] = self.pricing_tool_box.df_under_study['date_time'].dt.strftime('%HH-%MM')
        self.pricing_tool_box.df_under_study['date'] = self.pricing_tool_box.df_under_study['date_time'].dt.strftime('%d-%m')
        self.df_2h_reduced = self.pricing_tool_box.df_under_study[self.pricing_tool_box.df_under_study.loc[:, 'HH-MM'].isin(['06H-00M'
                                                                                    , '08H-00M'
                                                                                    , '10H-00M'
                                                                                    , '12H-00M'
                                                                                    , '14H-00M', '16H-00M', '18H-00M',
                                                                                 '20H-00M'
                                                                                    , '22H-00M', '00H-00M', '02H-00M',
                                                                                 '04H-00M'

                                                                                 ])].loc[:,
                  ('Close', 'Volume', 'date_time', 'day_of_week', 'HH-MM', 'date')].copy()
        # lets calculate the rolling metrics

        self.df_metrics = self.pricing_tool_box.calculate_rolling_metrics(self.df_2h_reduced, 'Close').copy()


    def use_correlogram_and_give_predictions(self, current_correlogram,curent_day_studied):
        """
        needs to be parametrized : not always 0.4

        lets take the latest returns in 24h per 2h tranches : df_metrics.tail(13)


        :param current_correlogram:
        :return:
        """
        self.Daystudied = curent_day_studied
        self.df_loaded_correlogram = current_correlogram
        print(self.df_loaded_correlogram)
        #self.load_2h_correlogram(choosen_day)
        #self.loaded_correlogram

        strat.df_loaded_correlogram = self.df_loaded_correlogram.iloc[1:, 1:].astype('float')

        df_result_corr_criterion = self.df_loaded_correlogram.where(
            (self.df_loaded_correlogram > 0.4) | (self.df_loaded_correlogram < -0.4))
        df_result_corr_criterion = df_result_corr_criterion.where(df_result_corr_criterion != 1.0)

        # this puts in form the desired pair of result, still duplicated :

        self.df_use_time_period = pd.DataFrame(columns=['hour_pair1', 'hour_pair2', 'corr'])
        for i in df_result_corr_criterion:
            for j in df_result_corr_criterion.columns:
                print(df_result_corr_criterion.loc[i, j])
                if df_result_corr_criterion.loc[i, j] > 0.4:
                    self.df_use_time_period.loc[i + '_' + j, 'hour_pair1'] = i
                    self.df_use_time_period.loc[i + '_' + j, 'hour_pair2'] = j
                    self.df_use_time_period.loc[i + '_' + j, 'corr'] = df_result_corr_criterion.loc[i, j]
                if df_result_corr_criterion.loc[i, j] < -0.4:
                    self.df_use_time_period.loc[i + '_' + j, 'hour_pair1'] = i
                    self.df_use_time_period.loc[i + '_' + j, 'hour_pair2'] = j
                    self.df_use_time_period.loc[i + '_' + j, 'corr'] = df_result_corr_criterion.loc[i, j]

        # self.df_use_time_period = self.df_use_time_period[self.df_use_time_period['corr']!=1]
        # let's remove the non complian (late correl) and keep the buy early sell later
        self.df_use_time_period.where(
            self.df_use_time_period['hour_pair1'] < self.df_use_time_period['hour_pair2']).dropna()
        self.df_use_time_period = self.df_use_time_period.reset_index()

        # a little bit of clean up for low history dataframes
        self.df_use_time_period = self.df_use_time_period[self.df_use_time_period['corr'] > -0.97]
        self.df_use_time_period = self.df_use_time_period[self.df_use_time_period['corr'] < 0.97]
        # we reduce latest returns to the studied days

        # Ya un problem , ca devrait etre la derbneiere date


        print(curent_day_studied,self.Daystudied)
        self.df_latest_returns =self.df_metrics.copy()
        self.df_latest_returns  =self.df_latest_returns[self.df_latest_returns['day_of_week'] == self.Daystudied]
        self.df_latest_returns = self.df_latest_returns.tail(13)

        """
        this produces : 
                    index                hour_pair1 hour_pair2     corr
                    0  06H-00M_12H-00M    06H-00M    12H-00M  0.46154
                    1  06H-00M_18H-00M    06H-00M    18H-00M -0.42795
                    2  06H-00M_22H-00M    06H-00M    22H-00M -0.41919
                    4  12H-00M_22H-00M    12H-00M    22H-00M -0.44836
                    5  14H-00M_18H-00M    14H-00M    18H-00M -0.40731

        """
        #this provides us with the correct time frames to use in the "predict phase"
        # to be enhanced with predict power
        #
        self.df_identified_opportunities = 'no dataframe created'
        #self.df_identified_opportunities = self.df_latest_returns.merge(right=self.df_use_time_period, how='outer',left_on='HH-MM', right_on='hour_pair1')

        try :
            self.df_identified_opportunities = self.df_latest_returns.merge(right=self.df_use_time_period, how='left', left_on='HH-MM', right_on='hour_pair1')
        except (IndexError,KeyError,ValueError):
            logger.info('==     No identified correlogram opportunities    ==================')

        # lets locate Buy and Sell opportunities - AMARCHE PAS

        self.df_identified_opportunities['predict'] = 'No Action'
        self.df_identified_opportunities['predict'] = np.where(self.df_identified_opportunities['log_returns'] * self.df_identified_opportunities['corr'] > 0.0, 'Buy _ open trade', 'Sell _ open trade')

        # lets make a rolling excel of opportunities , to be replaced by a database

        self.df_previous_opportunities = pd.read_excel('/home/simon/PythonWorkspace/Cistercian_bot/Opportunities_backlog/opportunities_backlog_correl.xlsx')

        self.df_opportunities_backlog= self.df_identified_opportunities.loc[:,('date_time', 'log_returns', 'corr', 'hour_pair1', 'hour_pair2', 'predict')]
        self.df_opportunities_backlog.loc[:,'status'] = 'not treated'

        self.df_opportunities_backlog = self.df_previous_opportunities.merge(self.df_opportunities_backlog,how='outer').copy()

        self.treat_opportunities_correl(current_datetime = self.current_datetime)

        self.df_opportunities_backlog.to_excel(params.parameters.fp_opportunity_correl,index=False)


        """
        le coeur du reacteur est la 
        rajouter un acces base pour logguer les opportunités TO DO 
        ou au pire un excel sauve sur le disque 
        """
    def treat_opportunities_correl(self,current_datetime):

        #1 determine the future time frame of the opportunity

        self.df_opportunities_backlog['pair1_tgt_datetime'] = self.df_opportunities_backlog['date_time'].dt.date
        self.df_opportunities_backlog['pair1_tgt_datetime_hour'] = pd.to_datetime(self.df_opportunities_backlog['hour_pair1'],
                                                                             format='%HH-%MM')
        self.df_opportunities_backlog['pair1_tgt_datetime_target'] = self.df_opportunities_backlog['date_time'].dt.date.astype('datetime64[ns]')
        self.df_opportunities_backlog['pair1_tgt_datetime_target'] += pd.to_timedelta(self.df_opportunities_backlog['pair1_tgt_datetime_hour'].dt.hour, unit='hours')


        self.df_opportunities_backlog['pair2_tgt_datetime'] = self.df_opportunities_backlog['date_time'].dt.date
        self.df_opportunities_backlog['pair2_tgt_datetime_hour'] = pd.to_datetime(self.df_opportunities_backlog['hour_pair2'],
                                                                             format='%HH-%MM')
        self.df_opportunities_backlog['pair2_tgt_datetime_target'] = self.df_opportunities_backlog['date_time'].dt.date.astype('datetime64[ns]')
        self.df_opportunities_backlog['pair2_tgt_datetime_target'] += pd.to_timedelta(self.df_opportunities_backlog['pair2_tgt_datetime_hour'].dt.hour, unit='hours')



        # 2 determine the expected return

        self.df_opportunities_backlog['expected_return'] = self.df_opportunities_backlog['corr'] * self.df_opportunities_backlog['log_returns']

        self.df_opportunities_backlog['current_pricing_date'] = current_datetime
        self.last_opportunities_update = self.df_opportunities_backlog.loc[:,'date_time'].tail(1).iloc[0]

        #3 -
        # update the lists of trades to be done for the given time period

        for i in self.df_opportunities_backlog.index:
            if self.df_opportunities_backlog.loc[i,'status'] == 'not treated':
            # case of new opportunity rising
                if self.df_opportunities_backlog.loc[i,'expected_return'] > 0.0:
                    self.df_opportunities_backlog.loc[i, 'opportunity_current'] = 'Buy _ open trade'
                    self.df_opportunities_backlog.loc[i, 'opportunity_open'] = 'Buy _ open trade'
                    self.df_opportunities_backlog.loc[i, 'opportunity_close'] = 'Sell _ close trade'
                    self.df_opportunities_backlog.loc[i, 'status'] = 'ongoing'
                elif self.df_opportunities_backlog.loc[i,'expected_return'] < 0.0:
                    self.df_opportunities_backlog.loc[i, 'opportunity_current'] = 'Sell _ open trade'
                    self.df_opportunities_backlog.loc[i, 'opportunity_open'] = 'Sell _ open trade'
                    self.df_opportunities_backlog.loc[i, 'opportunity_close'] = 'Buy _ close trade'
                    self.df_opportunities_backlog.loc[i, 'status'] = 'ongoing'
            elif self.df_opportunities_backlog.loc[i,'status'] == 'ongoing':
            # case of closing ongoing opportunities
                if current_datetime > self.df_opportunities_backlog.loc[i,'pair2_tgt_datetime_target']:
                    self.df_opportunities_backlog.loc[i, 'opportunity_current'] = self.df_opportunities_backlog.loc[i, 'opportunity_close']


        # 4 for a split period ; if there is trades in between , I want to split individual periods in individual trades
        # eg if I have a +2 on 00H to 08h , and a -1 on 04H to 06H , I want to split in 00H - 04H , 4h-6H and 06h -> 08H

        # 4.1 let's firast correct the duplicates in the backlog (join to be corrected first)
        self.df_opportunities_backlog = self.df_opportunities_backlog.groupby(by=['date_time', 'pair2_tgt_datetime_target', 'pair1_tgt_datetime_target']).max()
        self.df_opportunities_backlog.reset_index(inplace=True)
        self.df_opportunities_backlog.loc[:, 'updated_pair1_tgt_datetime_target'] = self.df_opportunities_backlog.loc[:, 'pair1_tgt_datetime_target'].copy()

        # 4.2 then produce the updated pair 1 tgt
        for i in self.df_opportunities_backlog.index:
            try:
                if (self.df_opportunities_backlog.loc[i, 'hour_pair1'] == self.df_opportunities_backlog.iloc[self.df_opportunities_backlog.index.get_loc(i - 1)]['hour_pair1']) \
                        & (self.df_opportunities_backlog.loc[i, 'pair2_tgt_datetime_target'] > self.df_opportunities_backlog.loc[i, 'pair1_tgt_datetime_target'])\
                        & (self.df_opportunities_backlog.loc[i, 'status'] == 'ongoing'):
                    self.df_opportunities_backlog.loc[i, 'updated_pair1_tgt_datetime_target'] = self.df_opportunities_backlog.iloc[self.df_opportunities_backlog.index.get_loc(i - 1)]['pair2_tgt_datetime_target']
            except (KeyError):
                logger.info('==     No identified updated target time 1 '+ str(i) + ')    ==================')
                pass


        # 99 update the status

        # 99.1 close all out dated opportunities
        self.df_opportunities_backlog.loc[:,'status'] = np.where(self.df_opportunities_backlog.loc[:,'pair1_tgt_datetime_target'] < current_datetime,'to be closed',self.df_opportunities_backlog.loc[:,'status'])


    def define_what_current_2h_period_is_next(self,latest_df):
        """
        takes a df as input , returns the next 2h period
        :param latest_df:
        :return:
        """

        latest_df['hour'] = latest_df['date_time'].dt.hour


        if 0<=latest_df['hour'].tail(1).iloc[0]<=2:
            return('02H-00M_04H-00M')
        elif 2 <= latest_df['hour'].tail(1).iloc[0] < 4:
            return ('04H-00M_06H-00M')
        elif 4 <= latest_df['hour'].tail(1).iloc[0] < 6:
            return ('06H-00M_08H-00M')
        elif 6 <= latest_df['hour'].tail(1).iloc[0] < 8:
            return ('08H-00M_10H-00M')
        elif 8 <= latest_df['hour'].tail(1).iloc[0] < 10:
            return ('10H-00M_12H-00M')
        elif 10 <= latest_df['hour'].tail(1).iloc[0] < 12:
            return ('12H-00M_14H-00M')
        elif 12 <= latest_df['hour'].tail(1).iloc[0] < 14:
            return ('14H-00M_16H-00M')
        elif 14 <= latest_df['hour'].tail(1).iloc[0] < 16:
            return ('16H-00M_18H-00M')
        elif 16 <= latest_df['hour'].tail(1).iloc[0] < 18:
            return ('18H-00M_20H-00M')
        elif 18 <= latest_df['hour'].tail(1).iloc[0] < 20:
            return ('20H-00M_22H-00M')
        elif 20 <= latest_df['hour'].tail(1).iloc[0] < 22:
            return ('22H-00M_00H-00M')
        else:
            return('No period')

    def define_what_current_2h_period(self,dt_hour):
        """
        dt_hour must be in date time . time() format

        takes a df as input , returns the next 2h period
        :param latest_df:
        :return:

        u.time() example :  datetime.time(16, 53, 16, 801938)
        must return 16H-00M_18H-00M'

        """

        if datetime.time(0,0,0)<=dt_hour<datetime.time(2,0,0):
            return('00H-00M_02H-00M')
        elif datetime.time(2, 0, 0) <= dt_hour < datetime.time(4, 0, 0):
                return ('02H-00M_04H-00M')
        elif datetime.time(4, 0, 0) <= dt_hour < datetime.time(6, 0, 0):
                return ('04H-00M_06H-00M')
        elif datetime.time(6, 0, 0) <= dt_hour < datetime.time(8, 0, 0):
                return ('06H-00M_08H-00M')
        elif datetime.time(8, 0, 0) <= dt_hour < datetime.time(10, 0, 0):
                return ('08H-00M_10H-00M')
        elif datetime.time(10, 0, 0) <= dt_hour < datetime.time(12, 0, 0):
                return ('10H-00M_12H-00M')
        elif datetime.time(12, 0, 0) <= dt_hour < datetime.time(14, 0, 0):
                return ('12H-00M_14H-00M')
        elif datetime.time(14, 0, 0) <= dt_hour < datetime.time(16, 0, 0):
                return ('14H-00M_16H-00M')
        elif datetime.time(16, 0, 0) <= dt_hour < datetime.time(18, 0, 0):
                return ('16H-00M_18H-00M')
        elif datetime.time(18, 0, 0) <= dt_hour < datetime.time(20, 0, 0):
                return ('18H-00M_20H-00M')
        elif datetime.time(20, 0, 0) <= dt_hour < datetime.time(22, 0, 0):
                return ('20H-00M_22H-00M')
        elif datetime.time(22, 0, 0) <= dt_hour < datetime.time(0, 0, 0):
                return ('22H-00M_00H-00M')
        else:
            return('No period')

    def set_2H_correlation_criterion(self):
        self.name = 'set_2H_correlation_'

        self.df_bt_15m = self.dfstrat_understudy.copy()
        #self.df_bt_15m.tail().loc[:, 'date_time']

        self.df_bt_15m['day_of_week'] = self.df_bt_15m['date_time'].dt.day_name()
        self.df_bt_15m['HH-MM'] = self.df_bt_15m['date_time'].dt.strftime('%HH-%MM')
        self.df_bt_15m['date'] = self.df_bt_15m['date_time'].dt.strftime('%d-%m')
        df = self.df_bt_15m[self.df_bt_15m.loc[:, 'HH-MM'].isin(['06H-00M'
                                                          , '08H-00M'
                                                          , '10H-00M'
                                                          , '12H-00M'
                                                          , '14H-00M', '16H-00M', '18H-00M', '20H-00M'
                                                          , '22H-00M', '00H-00M', '02H-00M', '04H-00M'

                                                       ])].loc[:,
             ('Close', 'Volume', 'date_time', 'day_of_week', 'HH-MM', 'date')].copy()
        obj_test = strat.deribit_algo_strat()
        self.df_metrics = obj_test.calculate_rolling_metrics(df, 'Close')
        if self.tail_limit=='NO':
            pass
        else : 
            self.df_metrics = self.df_metrics.tail(self.tail_limit).copy()
            


        #let's take the current day as day of study
        try:
            self.Daystudied = self.df_metrics.loc[:,'day_of_week'].tail(1).iloc[0]
            print('day studied is : ',self.Daystudied)
            logging.info(self.name,'day studied', self.Daystudied)
        except IndexError:
            self.Daystudied ='error'
            logging.error(self.name,'day studied', self.Daystudied)
            pass


        try:
            self.df_fridays = self.df_metrics[self.df_metrics.loc[:, 'day_of_week'] == self.Daystudied].loc[:,
                     ('log_returns', 'HH-MM', 'date')].copy()
        except IndexError:
            pass
        try:
            new_df = self.df_fridays[self.df_fridays['date'] == self.df_fridays['date'].unique()[0]].copy()
            new_df = new_df.rename({'log_returns': 'log_returns' + str(self.df_fridays['date'].unique()[0])}, axis=1)
        except IndexError :
            return('no_modifs')
            pass
        del (new_df['date'])
        for i in self.df_fridays['date'].unique()[1:]:
            print(i)
            new_df = pd.merge(left=new_df,
                              right=self.df_fridays[self.df_fridays['date'] == i],
                              how="outer",
                              on='HH-MM',
                              left_index=False,
                              right_index=False,
                              sort=False,
                              suffixes=("_x", "_y"),
                              copy=False,
                              indicator=False,
                              validate=None)
            new_df = new_df.rename({'log_returns': 'log_returns' + str(i)}, axis=1)
            del (new_df['date'])

        new_df.set_index('HH-MM', inplace=True)
        self.res = new_df.transpose(copy=True)
        self.res = self.res.loc[:, ~self.res.columns.duplicated()]
        self.df_corr = self.res.corr()
        figsize = (300, 225)

        """
        # sns.heatmap(df_corr,cmap="YlGnBu",title=Daystudied)
        bx = sns.heatmap(self.df_corr, cmap="Blues", linewidth=0.3, annot=True, fmt=".2f", cbar_kws={"shrink": .8})
        bx.set_title(self.Daystudied)
        plt.show()
        """
        rows_to_look_at = self.df_corr.where((self.df_corr > self.pos_corr_criterion) & (self.df_corr != 1.0)).dropna(axis=0,
                                                                                                     how='all').index
        logging.info(str(self.name),'this are the correl to look at : ', str(rows_to_look_at))

        # this confirms that we are in the good direction :
        if (len(rows_to_look_at)==2) and (self.df_corr.loc[rows_to_look_at[0], rows_to_look_at[1]] > self.pos_corr_criterion):
            print('yippekaï pauvre con')
            if (self.df_metrics.loc[:,'HH-MM'].tail(1)==rows_to_look_at[0]).iloc[0] :
                # we are in the right day and right timeschudule
                if (self.df_metrics.tail(1).loc[:,'log_returns'] > 0).iloc[0] :
                    #move up - the n buy
                    logger.info('Buy open trade returned')
                    return('Buy _ open trade')
                if (self.df_metrics.tail(1).loc[:, 'log_returns'] < 0).iloc[0]:
                    logger.info('Sell open trade returned')
                    return('Sell_open trade ')

                if (self.df_metrics.loc[:, 'HH-MM'].tail(1) == rows_to_look_at[1]).iloc[0]:

                    if (self.df_metrics.tail(1).loc[:, 'log_returns'] < 0).iloc[0]:
                        # move up - the n buy
                        logger.info('Sell close trade returned')
                        return ('Sell _ close previous open trade')
                    if (self.df_metrics.tail(1).loc[:, 'log_returns'] > 0).iloc[0]:
                        logger.info('Buy Close trade returned')
                        return ('Buy _close previous open  trade ')

        # need to update on good steady correlation selection

        #we found 0.7 between 02h and 16h - so lets use it

        return('-')

    def simple_strat_for_testing(self):

        self.df_bt_15m = self.dfstrat_understudy.copy()
        if self.df_bt_15m.loc[:,'log_returns_ewma_std_dev'].tail(1).iloc[0]<0.0030:
            logger.info('Buy open trade returned')
            return('Buy _ open trade')
        
        elif self.df_bt_15m.loc[:,'log_returns_ewma_std_dev'].tail(1).iloc[0]>0.0080:
            logger.info('Sell Close trade returned')
            return('Sell _ Close trade')
        
        else:
            logger.info('no strat action ')
            return('do nothing')

    def simple_strategy_vectorized(self):
        """:cvar
        this is not a method
        you want me to go back to my plane to france ?

        """
        self.dfstrat_understudy
        obj_test = strat.deribit_algo_strat()
        df_metrics = obj_test.calculate_rolling_metrics(self.dfstrat_understudy, 'Close')






class backtest_crypto_strategy_simple(another_set_of_metrics, another_algo_strategy):
    """
    this class allows to set a simple strategy by taking measures and set a buy or a sell on given instruments
    measures : takes metrics on a given market and pair
    trigger : set a metric level to perform a market action


    1°) loads a backtest file (can be large )
    2°) sets strategy params
    3°) performs backtest (can be agnostic)
        in a loop hole;


    """

    def __init__(self, name, df_under_study):
        super().__init__(name, df_under_study)
        self.name = name
        self._all_path = ['E:/cryptos/binance_histo/24052021-09_051dbt_btcusdt_1D.xlsx',
                          'E:/cryptos/binance_histo/latest_backtest_btcusdt.xlsx',
                          '/home/simon/PythonWorkspace/Cistercian_bot/backtest_ressource/01082021-17_08_fq_1dbt_btcusdt.xlsx',
                          '/home/simon/PythonWorkspace/Cistercian_bot/backtest_ressource/05082021-18_08_fq_15mbt_btcusdt.xlsx'
                          ]


    def load_backtest_scenario(self,frequency='15mn'):

        if frequency == '15mn':

            self.filepath = self._all_path[3]
            self.df_backtest = pd.read_excel(self.filepath)
            self.df_under_study = self.df_backtest.copy()
            logger.info('===== local backtest 15mn loaded ====')

        else :
            #loads backtest only if not loaded
            self.filepath = self._all_path[2]
            self.df_backtest = pd.read_excel(self.filepath)
            self.df_under_study = self.df_backtest.copy()
            logger.info('===== local backtest 1d loaded ====')

    def set_strategy_vol_params(self):
        """

        Buy_signal : ewma_std  < 2*min
        Sell_signal  ewms_std > max/2

        """
        self.measure_rolling_ewma_std()
        self.underlying_instrument = 'Close'

        #np.where(df.loc[:,'log_returns_ewma_std_dev']>0.03,True,False)
        name = 'vol_params'
        self.strat1_buy_signal = name+'buy_signal'
        self.strat1_sell_signal = name+'sell_signal'

        self.df_backtest.loc[:,name+'buy_signal'] = np.where(self.df_backtest.loc[:,self.metric1['metric_name']]< 6*self.metric1['minimum'], True, False)
        self.df_backtest.loc[:,name+'buy_signal'] = np.where(self.df_backtest.loc[:,'log_returns']<-0.05, True, False)
        self.df_backtest.loc[:,name+'sell_signal'] = np.where(self.df_backtest.loc[:,self.metric1['metric_name']] > self.metric1['maximum']/5.5, True, False)
        self.df_backtest.loc[:,name+'sell_signal'] = np.where(self.df_backtest.loc[:,'Close_ewma_std_dev'] > 500, True, False)

    def allocation_function(self,style):

        if style=='full':
            self.allocation_ratio = 1.0
        else:
            pass


    def perform_backtest(self):
        """
        to be updated with ledger management... in case of short selling or repo
        :cvar
        simple strat all of nothing
        """
        self.set_strategy_vol_params()

        self.portfolio_cash_amount_usdt = 0.5*10223.08
        eq_usdt_amount = self.portfolio_cash_amount_usdt
        self.portfolio_cash_amount_btc = 0.0
        self.cash_remaining=1.0
        self.btc_remaining=0.0





        for i in self.df_backtest.index:
            self.df_backtest.loc[i, 'eq_usdt_amount'] = eq_usdt_amount
            eq_usdt_amount = self.portfolio_cash_amount_btc * self.df_backtest.loc[i, 'Close'] + self.portfolio_cash_amount_usdt

            if self.df_backtest.loc[i,self.strat1_buy_signal] :

                #let's see how much to allocate
                self.allocation_function(style='full')
                if self.portfolio_cash_amount_usdt != 0.0:
                    # buy action
                    updated_btc_balance = (self.portfolio_cash_amount_usdt/self.df_backtest.loc[i, 'Close'])*self.allocation_ratio
                    updated_balance_usdt = 0.0
                    self.df_backtest.loc[i, 'buy_price'] = self.df_backtest.loc[i, 'Close']
                    self.df_backtest.loc[i, 'portfolio_amount_btc'],self.portfolio_cash_amount_btc = updated_btc_balance,updated_btc_balance
                    self.df_backtest.loc[i, 'portfolio_amount_usdt'],self.portfolio_cash_amount_usdt = updated_balance_usdt,updated_balance_usdt

                else:
                    pass

            elif self.df_backtest.loc[i, self.strat1_sell_signal]:
                if self.portfolio_cash_amount_btc != 0.0:
                    self.allocation_function(style='full')
                    self.df_backtest.loc[i, 'sell_price'] = self.df_backtest.loc[i, 'Close']
                    updated_balance_usdt = (self.portfolio_cash_amount_btc*self.df_backtest.loc[i, 'Close'])* self.allocation_ratio
                    updated_btc_balance = (1-self.allocation_ratio)
                    self.df_backtest.loc[
                        i, 'portfolio_amount_btc'], self.portfolio_cash_amount_btc = updated_btc_balance, updated_btc_balance
                    self.df_backtest.loc[
                        i, 'portfolio_amount_usdt'], self.portfolio_cash_amount_usdt = updated_balance_usdt, updated_balance_usdt


                else:
                    pass


            if i == self.df_backtest.index[-1]:
                #sell all - latest line back test
                updated_balance_usdt = (self.portfolio_cash_amount_btc*self.df_backtest.loc[i, 'Close'])* self.allocation_ratio
                updated_btc_balance = 0
                self.df_backtest.loc[
                    i, 'portfolio_amount_btc'], self.portfolio_cash_amount_btc = updated_btc_balance, updated_btc_balance
                self.df_backtest.loc[
                    i, 'portfolio_amount_usdt'], self.portfolio_cash_amount_usdt = updated_balance_usdt, updated_balance_usdt



        self.backtest_performance = self.df_backtest.loc[self.df_backtest.index[-1], 'eq_usdt_amount'] - self.df_backtest.loc[self.df_backtest.index[0], 'eq_usdt_amount']
        print ('la perf constatée est : ', self.backtest_performance)

        self.df_backtest.to_excel('C:/Data_repo/crypto/backtest_result/latest_backtest.xlsx')


    def perform_backtest_start_agnostic(self,mode='15mn',backtest_length=1000):
        # 7000 produces good results
        self.load_backtest_scenario(frequency=mode)
        self.selected_frequency = mode

        """vaste sujet que celui du backtest .. on part ici sur un aller / retour entre USDT (cash) et BTC (Asset) 
        # on fera les caculs de risque de contrepartie plus tard , en considérant que l'USDT est du cash sans risque (ce qui est faux) 
        # etapes : 
        1 - allocation du capital cash Vs BTC 
        2 - pour un dataetime donné, calcul des buy signal / sell signal 
        3 - Actions 
        4 - calcul de risk ? 
        5- next period ? 
        """

        'let s load  as strategy of correls '

        #let's start with a shorter backtest



        #main loop playing the time composante of the back test, putting the strat result in a column
        logger.info('=================starting backtest==================')


        # 0 - Inputs parameters

        self.set_strategy_vol_params()
        # lets start with 5000 Usd
        self.portfolio_cash_amount_usdt = 5000
        self.eq_usdt_amount = self.portfolio_cash_amount_usdt
        self.portfolio_cash_amount_btc = 0.0
        self.cash_remaining=1.0
        self.btc_remaining=0.0

        self.df_under_study = self.df_under_study.tail(backtest_length).copy()
        self.df_under_study['day_of_week'] = self.df_under_study['date_time'].dt.day_name()

        self.df_under_study['HH-MM'] = self.df_under_study['date_time'].dt.strftime('%HH-%MM')



        for dt in self.df_under_study.index:
            # this is the loop to play the strategy over the progression of time : dt
            # iterates each time by recreating current df
            # ===> please write outputs in self.df_under_study !!!
            self.current_df_backtest = self.df_under_study.loc[:dt,:].copy()
            print('backtest tail : ', self.current_df_backtest.tail(1), 'dt', dt, 'length', len(self.current_df_backtest))
            logger.info('================current dt in backtest : ',str(dt),'=====================')

            """
            First lets have a working backtest on simple strat Vol > 0.3 buy , vol < 0.1 , Sell 
            then we will try to have a correl : 
            self.choosen_strat = another_algo_strategy(strategy_name='correl_2H_blocs',dfstrat_under_study=current_df_backtest)
            self.df_under_study.loc[dt,'strat_result'] = self.choosen_strat.set_2H_correlation_criterion()
         
            
            """

            # 1 - for the current known histo (up to current df back trst, play the strat

            self.choosen_strat = another_algo_strategy(strategy_name= 'testing backtest',dfstrat_under_study = self.current_df_backtest.copy())

            # prepare strategy parameters
            self.current_day_in_backtest = self.current_df_backtest['day_of_week'].tail(1).iloc[0]
            self.current_2h_time_period = self.current_df_backtest['HH-MM'].tail(1).iloc[0]
            self.choosen_strat.df_bt_15m = self.current_df_backtest.copy()
            self.choosen_strat.calculate_2h_correl(self.current_day_in_backtest)
            try :
                print(self.choosen_strat.df_corr)
                logger.info('================correl calclulated  : ',str(self.choosen_strat.df_corr), '=====================')
            except AttributeError:
                try:
                    logger.info('================ NO correl calclulated  : ',str(self.choosen_strat.df_corr),
                                '=====================')
                except AttributeError:
                    pass

                pass

            try:
                self.df_under_study.loc[dt,'next2h_timeperiod'] = self.choosen_strat.define_what_current_2h_period_is_next(self.current_df_backtest.tail(5))
            except(KeyError,AttributeError,IndexError):
                print('no functionna')


            try :

                self.df_under_study.loc[dt, 'correlogram'] = str(self.choosen_strat.df_corr)
                self.choosen_strat.use_correlogram_and_give_predictions(self.choosen_strat.df_corr,self.current_2h_time_period)


            except (KeyError,AttributeError,IndexError):
                logger.info('================ NO correl calclulated =====================')
                pass

            try:
                self.df_under_study.loc[dt, 'opportunities'] = str(self.choosen_strat.df_identified_opportunities)

                self.df_under_study.loc[dt, 'strat_result'] = self.choosen_strat.df_identified_opportunities.tail(1).loc[:,'predict']
            except (ValueError,AttributeError):
                pass



            #self.df_under_study.loc[dt, 'strat_result'] = self.choosen_strat.simple_strat_for_testing()
            # If Buy & cash , then Buy
            # if Sell and BTC then Sell
            #If anything else , then Rien



        # second loop to calculate the buy / sell and the gains

        for i in self.df_under_study.index:
            if self.df_under_study.loc[i,'strat_result'] == 'Buy _ open trade':
                # opening a buy position

                #let's see how much to allocate
                self.allocation_function(style='full')
                if self.portfolio_cash_amount_usdt != 0.0:
                    # buy action if there is cash to allocate
                    updated_btc_balance = (self.portfolio_cash_amount_usdt/self.df_backtest.loc[i, 'Close'])*self.allocation_ratio
                    updated_balance_usdt = 0.0
                    self.df_backtest.loc[i, 'buy_price'] = self.df_backtest.loc[i, 'Close']
                    logger.info('=====*** buy action @', str(self.df_backtest.loc[i, 'Close']), '****=====')
                    self.df_backtest.loc[i, 'portfolio_amount_btc'],self.portfolio_cash_amount_btc = updated_btc_balance,updated_btc_balance
                    self.df_backtest.loc[i, 'portfolio_amount_usdt'],self.portfolio_cash_amount_usdt = updated_balance_usdt,updated_balance_usdt
                else:
                    logger.info('no money available for the buy')
                    pass

            elif self.df_under_study.loc[i,'strat_result'] == 'Sell _ Close trade':
                if self.portfolio_cash_amount_btc != 0.0:
                    self.allocation_function(style='full')
                    self.df_backtest.loc[i, 'sell_price'] = self.df_backtest.loc[i, 'Close']
                    logger.info('=====*** Sell action closing position @', str(self.df_backtest.loc[i, 'Close']), '****=====')

                    updated_balance_usdt = (self.portfolio_cash_amount_btc * self.df_backtest.loc[
                        i, 'Close']) * self.allocation_ratio
                    updated_btc_balance = (1 - self.allocation_ratio)
                    self.df_backtest.loc[
                        i, 'portfolio_amount_btc'], self.portfolio_cash_amount_btc = updated_btc_balance, updated_btc_balance
                    self.df_backtest.loc[
                        i, 'portfolio_amount_usdt'], self.portfolio_cash_amount_usdt = updated_balance_usdt, updated_balance_usdt

                else:
                    logger.info('no BTC  available for the Selling ')
                    pass





        # if success , adds entet and suffix and save the results
        logger.info('=================End backtest : SUCCESS ==================')
        path_save_backtest = '/Cistercian_bot/backtest_ressource/results/'
        now = datetime.datetime.now()
        entete = 'cis_zozz_'
        prefix = str(now.strftime('%d-%m-%Y_%H_%M'))
        filename_backtest_result = 'estabaq_bt_result'
        xls_suffix = '.xlsx'

        self.df_under_study.to_excel(path_save_backtest+entete+prefix+filename_backtest_result+xls_suffix)
        logger.info('=================End backtest : SUCCESS :: Output to :  ==================',path_save_backtest+entete+prefix+filename_backtest_result)


    def get_data(self):
        self.load_backtest_scenario()
        self.df_backtest_reference = self.df_backtest.copy()

    def shorten_backtest(self,length=1200):
        self.df_backtest_reference = self.df_backtest.tail(length).copy()



    def run_strategy(self,shorten_backtest=True):
        """
        as stated by Y Hillpisch "python for algorithmytmic trading"
        yet , the strategy is not outsourced

        :return:
        """

        # 1 - Load BT
        self.get_data()
        if shorten_backtest:
            self.shorten_backtest()


        # 1' - Calculate metrics1


        #no need yet,to be done

        #2 - measure current strategy metrics
        self.df_backtest_reference.loc[:, 'log_returns_ewma_std_dev_local_min'] = self.df_backtest_reference.loc[:, 'log_returns_ewma_std_dev'].rolling(window=10).min()
        self.df_backtest_reference.loc[:, 'log_returns_ewma_std_dev_local_max'] = self.df_backtest_reference.loc[:,'log_returns_ewma_std_dev'].rolling(window=10).max()



        # play strategy per column

        # sells
        self.df_backtest_reference.loc[:, 'position'] = np.where(self.df_backtest_reference['log_returns_ewma_std_dev']<self.df_backtest_reference['log_returns_ewma_std_dev_local_min'], -1,1)
        self.df_backtest_reference.loc[:, 'position'] = np.where(
            self.df_backtest_reference['log_returns_ewma_std_dev'] > 0.8*self.df_backtest_reference[
                'log_returns_ewma_std_dev_local_max'], 1, -1)

        # self.df_backtest_short.dropna(inplace=True)

        # Buy Sell
        self.df_backtest_reference.loc[:, 'strategy'] = self.df_backtest_reference['position'].shift(1) * self.df_backtest_reference['log_returns']
        # calc returns
        self.df_backtest_reference['creturns'] = self.df_backtest_reference['log_returns'].cumsum().apply(np.exp)
        self.df_backtest_reference['cstrategy'] = self.df_backtest_reference['strategy'].cumsum().apply(np.exp)
        # summarize strategy
        self.results = self.df_backtest_reference
        # gross performance of the strategy
        self.aperf = self.df_backtest_reference['cstrategy'].iloc[-1]
        # out / -  underperformance of the strategy
        self.operf = self.aperf - self.df_backtest_reference['creturns'].iloc[-1]

        self.df_backtest_reference.loc[:, ('date_time', 'Close', 'cstrategy')].plot(x='date_time',
                                                                                            y=['Close', 'cstrategy'],
                                                                                            secondary_y='cstrategy')

        # if success , adds entet and suffix and save the results
        logger.info('=================End backtest : SUCCESS ==================')
        path_save_backtest = 'E:/cryptos/backtesting_result/'
        now = datetime.datetime.now()
        entete = 'cis_zozz_'
        prefix = str(now.strftime('%d-%m-%Y_%H_%M'))
        filename_backtest_result = 'estabaq_bt_result'
        xls_suffix = '.xlsx'

        self.df_backtest_reference.to_excel(path_save_backtest + entete + prefix + filename_backtest_result + xls_suffix)
        logger.info('=================End backtest : SUCCESS :: Output to :  ==================',
                    path_save_backtest + entete + prefix + filename_backtest_result)


        self.plot_results()

        return round(self.aperf, 2), round(self.operf, 2)

    def plot_results(self):
        """
        plots the cumulative performance


        :return:
        """

        if self.results is None:
            print('no results so far, please do run a strategy  ')
        title = 'strategy : ' + self.name
        self.results[['creturns','cstrategy']].plot(title=title,figsize=(10,6))
        self.df_backtest_reference.loc[:, ('date_time', 'Close', 'log_returns_ewma_std_dev')].plot(title=title,
            x='date_time', y=['Close', 'cstrategy'], secondary_y='cstrategy')






if __name__=='__main__':
  
    
    current_backtest = backtest_crypto_strategy_simple('test','test')
    current_backtest.perform_backtest_start_agnostic()
    #current_backtest.run_strategy(shorten_backtest=True)
    resulting_bactest = current_backtest.choosen_strat.df_opportunities_backlog.copy()

    resulting_bactest = resulting_bactest.merge(current_backtest.df_backtest.loc[:,('date_time','Close')],left_on='updated_pair1_tgt_datetime_target',right_on='date_time',how='left').copy()
    resulting_bactest.rename(columns={'Close':'btc_price_pair1'},inplace=True)
    
    resulting_bactest = resulting_bactest.merge(current_backtest.df_backtest.loc[:,('date_time','Close')],right_on='date_time',left_on='pair2_tgt_datetime_target',how='left').copy()
    resulting_bactest.rename(columns={'Close':'btc_price_pair2'},inplace=True)
    resulting_bactest['btc_difference_1-2'] = resulting_bactest['btc_price_pair2'] - resulting_bactest['btc_price_pair1']
    resulting_bactest['btc_benefice'] = np.where(resulting_bactest['opportunity_open']=='Buy _ open trade',resulting_bactest['btc_difference_1-2'],-resulting_bactest['btc_difference_1-2'])
    resulting_bactest['cum_btc_benefice'] = resulting_bactest['btc_benefice'].cumsum()
    resulting_bactest.loc[:,('date_time_x','cum_btc_benefice')].plot(x='date_time_x')





