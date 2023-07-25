# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:40:32 2019

@author: rr5862
"""
import numpy as np
import scipy.stats as si

import datetime
import datetime as dt
import logging
#import market

from math import sqrt, exp, log, pi
from scipy.stats import norm
import scipy.optimize as optim

"""
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
file_handler = RotatingFileHandler('C:/PythonWorkspace/pythonws_engie/owomodo_scrapping/config/logs/activity.log', 'a', 1000000, 1)
# on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
# créé précédement et on ajoute ce handler au logger
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# création d'un second handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)
"""
class instrument:
    def __init__(self,debug):
        self.debug = debug

    def set_debug_on(self):
        self.debug=True


class fixed_income(instrument):
    def __init__(self,debug,notional):
        super().__init__(debug)
        self.notional = notional


class fx_instrument(fixed_income):
    def __init__(self,debug,notional):
        super().__init__(debug,notional)
        self.domestic_currency ='USD'
        self.trade_date = '24012019'
        self.maturity_date = '25022019'

class fx_option(fx_instrument):
    def __init__(self,debug,notional):
        super().__init__(debug, notional)
        self.way = 'buy'
        self.type = 'call'
        self.trade_date = '24012019'
        self.maturity_date='25022019'
        self.strike = 1.1314
        self.volatility_ATM_1M = 0.067
        self.foreign_currency = 'GBP'
        self.last_valuation_date = False

    def set_option_characteristics(self,
                                   new_way='buy',
                                   new_type='call',
                                   new_foreign_currency='EUR',
                                   new_trade_date='24012019',
                                   new_maturity_date='25022019',
                                   new_strike=1.1315,
                                   new_volatility_ATM_1M=0.0617):
        self.way = new_way
        self.type = new_type
        self.strike = new_strike
        self.foreign_currency = new_foreign_currency
        self.trade_date = new_trade_date
        self.maturity_date = new_maturity_date
        self.volatility_ATM_1M = new_volatility_ATM_1M
        #logger.info('fxo_config updated')
        return 'Done'

    def date_conversion(self):
        self.trade_date = dt.datetime.strptime(self.trade_date,'%d%m%Y')
        self.maturity_date = dt.datetime.strptime(self.maturity_date,'%d%m%Y')
        try:
            self.last_valuation_date = dt.datetime.strptime(self.last_valuation_date, '%d%m%Y')
        except (TypeError,NameError):
            pass
    def calculate_discount_factor(self,ccy):
        maturity = '1M'
        current_df = 'DF_'+str(ccy)+'_'+maturity

        #current_curve = market.generate_curve_on_currency(ccy)
        #current_discount_factor = np.exp(-current_curve[maturity]*self.tau.total_seconds()/(360*24*3600)) #act360 convention
        #return [{current_df: current_discount_factor},current_curve]

    def option_pricer(self, valuation_date='24012019'):
        self.last_valuation_date = valuation_date
        self.date_conversion()
        self.tau = self.maturity_date - self.last_valuation_date
        self.usd_discount = self.calculate_discount_factor('USD')
        phi_to_use = self.usd_discount[0]['DF_USD_1M']



        #r_usd = market.curve_USD['1M']

        #phi_usd = np.exp(-r_usd*self.tau.total_seconds()/(360*24*3600)) #act360 convention

class option_equity(fx_instrument):
    def __init__(self,debug,notional):
        super().__init__(debug, notional)
        self.way = 'buy'
        self.type = 'call'
        self.trade_date = '24012019'
        self.maturity_date='25022019'
        self.strike = 1.1314
        self.sigma = 0.3

        self.S = 100.0
        self.K = 100.0
        self.t = 30.0 / 365.0
        self.r = 0.01
        self.C0 = 2.3
        self.P0 = 2.30


    #   Function to calculate the values of d1 and d2 as well as the call
    #   price.  To extend to puts, one could just add a function that
    #   calculates the put price, or combine calls and puts into a single
    #   function that takes an argument specifying which type of contract one
    #   is dealing with.
    def d(self, sigma,S, K, r, t):
        self.sigma = sigma
        d1 = 1 / (self.sigma * sqrt(t)) * ( log(S/K) + (r + sigma**2/2) * t)
        d2 = d1 - self.sigma * sqrt(t)
        return d1, d2

    def calculate_call_price(self,sigma, S, K, r, t, d1, d2):
        call_price = norm.cdf(d1) * S - norm.cdf(d2) * K * exp(-r * t)
        return call_price

    def calculate_put_price(self,sigma, S, K, r, t, d1, d2):
        put_price = -norm.cdf(-d1) * S + norm.cdf(-d2) * K * exp(-r * t)
        return put_price
    def helper_statics(self,type,s,k,c0,p0,t,r):
        self.type = type
        self.S = s
        self.K = k
        self.C0 = c0
        self.P0 = p0
        self.t = t
        self.r = r


    def implicit_volatility_bs(self):
        """
        
        not used anymore
        
        """
        #  Tolerances
        tol = 1e-3
        epsilon = 1
        #  Variables to log and manage number of iterations
        count = 0
        max_iter = 1000
        #  We need to provide an initial guess for the root of our function
        vol = 0.50
        while epsilon > tol:
            #  Count how many iterations and make sure while loop doesn't run away
            count += 1
            if count >= max_iter:
                print('Breaking on count')
                break;

            if abs(vol) >= 8.0:
                print('no converge')
                return 0.0

            #  Log the value previously calculated to computer percent change
            #  between iterations
            orig_vol = vol

            #  Calculate the vale of the call price
            d1, d2 = self.d(vol, self.S, self.K, self.r, self.t)
            if self.type == 'call':
                function_value = self.calculate_call_price(vol, self.S, self.K, self.r, self.t, d1, d2) - self.C0
            elif self.type =='put':
                function_value = self.calculate_put_price(vol, self.S, self.K, self.r, self.t, d1, d2) - self.C0

            #  Calculate vega, the derivative of the price with respect to
            #  volatility
            vega = self.S * norm.pdf(d1) * sqrt(self.t)

            #  Update for value of the volatility
            vol = -function_value / vega + vol

            #  Check the percent change between current and last iteration
            epsilon = abs((vol - orig_vol) / orig_vol)
        print('sigma :',vol)
        print('Code took ', count, ' iterations')
        return vol
    
    def find_vol(self,sig):
        d1, d2 = self.d(sig, self.S, self.K, self.r, self.t)
        if self.type == 'call':
            res = self.C0 - self.calculate_call_price(sig, self.S, self.K, self.r, self.t, d1, d2)
        elif self.type == 'put':
            res = self.P0 - self.calculate_put_price(sig, self.S, self.K, self.r, self.t, d1, d2)
        return res**2
    
    def compute_brent_vol(self):
        self.brent = optim.brent(self.find_vol,brack=(0.000001,3),full_output=True)
        #self.mini = optim.minimize_scalar(self.find_vol,bracket=(0.000001,3))
        #self.mini2 = optim.minimize(self.find_vol,[0.00001,2],method='CG',args=2.3)

    
    
    def vega(self,S, K, T, r, sigma):
        #from : https://aaronschlegel.me/implied-volatility-functions-python.html
        #S: spot price
        #K: strike price
        #T: time to maturity
        #r: interest rate
        #sigma: volatility of underlying asset
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * si.norm.cdf(d1, 0.0, 1.0) * np.sqrt(T)     
        return vega
    
    def expose_greeks(self):
        self.moneyness = self.S/self.K
 


if __name__ == '__main__':
    fi_inst = fixed_income(False,10000)
    fi_inst.debug
    opt = option_equity(True,100000)
    d1, d2 = opt.d(0.3, opt.S, opt.K, opt.r, opt.t)
    res = opt.calculate_call_price(0.3, opt.S, opt.K, opt.r, opt.t, d1, d2)
    
    
    """
    plt.plot(x,y,'b',label=r'$e^x-5*x$')
    plt.grid(True)
    plt.ylim( [-10,20])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper center')
    plt.show()
    """
