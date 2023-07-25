# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 20:18:30 2022

@author: Simon
"""
fp_correlogram = '/home/simon/PythonWorkspace/Cistercian_bot/backtest_ressource/Correlograms/'
correlogram_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
correl_tails = ['NO',4000,6000]


def return_day_name_from_weekday(isoweek_day=4):
    if isoweek_day == 0:
        return('Monday')
    elif isoweek_day == 1:
        return('Tuesday')
    elif isoweek_day == 2:
        return('Wednesday')
    elif isoweek_day == 3:
        return('Thursday')
    elif isoweek_day == 4:
        return('Friday')
    elif isoweek_day == 5:
        return('Saturday')
    elif isoweek_day == 6:
        return('Sunday')

def generate_fp_correl_example(day=correlogram_days[4],tail=correl_tails[1]):
    return(fp_correlogram+'corr__'+day+'Tail-'+str(tail)+'_full_correl.xlsx')

complete_correl_example = 'corr__WednesdayTail-4000_full_correl.xlsx'


fp_opportunity_correl = '/home/simon/PythonWorkspace/Cistercian_bot/Opportunities_backlog/opportunities_backlog_correl.xlsx'
