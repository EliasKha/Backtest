import library_instruments as lib
import pandas as pd
import sys
#import mysql.connector as sql
import datetime
from pytz import timezone 

row_limit = 50
query = """SELECT book_summary.*
    ,instruments.option_type
    ,instruments.strike
    ,instruments.contract_size
    ,instruments.base_currency
    ,instruments.quote_currency
    ,from_unixtime((instruments.expiration_timestamp)/1000) as expiration_ts
    ,instruments.taker_commission
FROM book_summary 
JOIN instruments  ON book_summary.instrument_name = instruments.instrument_name
where book_summary.implied_vol is null
order by book_summary.capture_ts asc
LIMIT """
query += str(row_limit)

query1 = """
select  book_summary.btc_price
	, book_summary.capture_ts
    ,book_summary.instrument_name
    ,instruments.option_type
   	,instruments.strike
    ,instruments.contract_size
    ,instruments.base_currency
    ,instruments.quote_currency
    ,from_unixtime((instruments.expiration_timestamp)/1000)
    ,instruments.taker_commission
    ,book_summary.*
FROM book_summary 
JOIN instruments  ON book_summary.instrument_name = instruments.instrument_name
where book_summary.implied_vol is null 
order by capture_ts desc
;
"""

query_emptylines = """
select count(*)from
book_summary 
JOIN instruments  ON book_summary.instrument_name = instruments.instrument_name
where book_summary.implied_vol is null
"""




#db_connection = sql.connect(host='localhost', database='shlomo', user='root', password='$$$Amelie')
#df_cases_to_calc = pd.read_sql(query, con=db_connection)

#print(df_cases_to_calc.head(5))
def execute_query_and_return_df(query_to_execute):
    """
    takes a query and executes it on shlomo
    :param query:
    :return:
    """
    db_connection = sql.connect(host='localhost', database='shlomo', user='root', password='$$$Amelie')
    # params to be added !
    df_to_analyse = pd.read_sql(query_to_execute, con=db_connection)
    return df_to_analyse


def plot_a_given_instrument(instrument_name='BTC-26JUN20-7000-P',start_date = '01/01/2020',end_date='25/04/2020'):
    """:arg
    df = graph.plot_a_given_instrument('BTC-26JUN20-7000-P','01/04/2020','30/06/2020')

    """

    query_prefix = """SELECT bs.id
        ,bs.btc_price
        ,bs.capture_ts
        ,bs.instrument_name
        ,bs.mark_price 
        ,bs.underlying_price
        ,bs.estimated_delivery_price
        ,bs.bid_price
        ,bs.ask_price
        ,bs.mid_price
        ,bs.volume
        ,bs.base_currency
        ,instruments.option_type
        ,instruments.strike
        ,instruments.contract_size
        ,instruments.base_currency
        ,instruments.quote_currency
        ,from_unixtime((instruments.expiration_timestamp)/1000)
        ,instruments.taker_commission
        #,bs.*
    FROM book_summary as bs
    JOIN instruments  ON bs.instrument_name = instruments.instrument_name
    where instruments.instrument_name ="""

    query_suffix = """order by bs.capture_ts asc"""
    query_to_execute = query_prefix+"'"+str(instrument_name)+"'"+query_suffix
    df_results = execute_query_and_return_df(query_to_execute)
    # time cutting

    df_results = df_results[df_results.loc[:, 'capture_ts'] > datetime.datetime.strptime(start_date, '%d/%m/%Y')].copy()
    df_results = df_results[df_results.loc[:, 'capture_ts'] < datetime.datetime.strptime(end_date, '%d/%m/%Y')].copy()
    #TO do : ajouter IV et volume
    #plotting
    plot = df_results.plot(x='capture_ts', y=['btc_price', 'underlying_price','mark_price'],secondary_y='mark_price',title=instrument_name)
    fig = plot.get_figure()
    fig.savefig('C:/cryptos_bot/research/last_fig.png')
    return df_results


class strategy():
    def ___init__(self):
        self.debug=False
        

def week_number_of_month(date_value):
     return (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)


def test_cheap_call_week1_expensive_week3():
    # get dataframe from history
    query = """SELECT book_summary.*
    ,instruments.option_type
    ,instruments.strike
    ,instruments.contract_size
    ,instruments.base_currency
    ,instruments.quote_currency
    ,from_unixtime((instruments.expiration_timestamp)/1000) as expiration_ts
        ,instruments.taker_commission
    FROM book_summary 
    JOIN instruments  ON book_summary.instrument_name = instruments.instrument_name
    where 1=1
    order by book_summary.capture_ts asc"""
    df_history_options = execute_query_and_return_df(query)
    df_history_options.loc[:,'next_future_expiry_cme'] = df_history_options.loc[:,'capture_ts'].apply(define_next_expiry_date_cme_btc_futures)
    
    df_history_options.loc[:,'time_diff_capture-expiry'] = df_history_options.loc[:,'next_future_expiry_cme'] - df_history_options.loc[:,'capture_ts']
    df_history_options.loc[:,'distance_from_K'] = abs(df_history_options.loc[:,'btc_price'] - df_history_options.loc[:,'strike'])
    df = df_history_options.copy()
    aa = df[(df.loc[:,'btc_price']==9775.61)&(df.loc[:,'option_type']=='call')]
    df_close_expi = df[(df.loc[:,'option_type']=='call')&(df.loc[:,'time_diff_capture-expiry']<datetime.timedelta(days=4))]
    
    # j'en suis resté la 
    
    return df_history_options
    
    # calculate which week of a month it is 
    # buy signal 
    # increment / decrement portfolio 
    # calc perfs
    

class research_indicators():
    def __init__():
        self.mode = 'research'
    def define_fwd_funnel(self):
        #A faire !!! 
        
        
        return False



def define_next_expiry_date_cme_btc_futures(date_inquired=datetime.datetime.now()):
    
    """
    as stated in 
    https://www.cmegroup.com/trading/equity-index/us-index/bitcoin_contractSpecs_options.html?optionProductId=8875#optionProductId=8875
    
    TERMINATION OF TRADING	
    Trading terminates at 4:00 p.m. London time 
    on the last Friday of the contract month. 
    
    If this is not both a London and U.S. business day, 
    trading terminates on the prior London and the U.S. business day.
    
    """
    
    london_tz = timezone('Europe/London')
    current_LDN_time = datetime.datetime.now(london_tz)
    
    try : 
        response = datetime.datetime(date_inquired.year,date_inquired.month,31)
    except ValueError:
        try: 
            response = datetime.datetime(date_inquired.year,date_inquired.month,30)
        except ValueError:
            response = datetime.datetime(date_inquired.year,date_inquired.month,28)
            
    while response.isoweekday() !=5:
        response = response-datetime.timedelta(days=1)
    
    
    # case with jour férié is missing !!! cal ? 
    
    return response
        
        

if __name__ == '__main__':
    res = research_indicators()
    res.define_fwd_funnel()

    
    
    

