# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 12:49:49 2021

@author: PC
"""

import asyncio
import websockets
import json

import nest_asyncio
nest_asyncio.apply()
__import__('IPython').embed()


"""
il va etre tres important de requeter toutes les grilles de prix par maturités , 
pour construire une nappe de vol et evaluer des prix d'option'
"""




msg = \
{
  "id" : 1,
  "method" : "public/get_mark_price_history",
  "params" : {
    "instrument_name" : "BTC-25JUN21-50000-C",
    "start_timestamp" : 1609376800000,
    "end_timestamp" : 1609376810000
  },
  "jsonrpc" : "2.0"
}

async def call_api(msg):
   async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
       await websocket.send(msg)
       while websocket.open:
           response = await websocket.recv()
           # do something with the response...
           print(response)

df = asyncio.get_event_loop().run_until_complete(call_api(json.dumps(msg)))


load instruments 

all_option_instruments = deribit.publicGetGetInstruments(params={'currency':'BTC','kind':'option','expired':'true'})

df_instument = pd.DataFrame(all_option_instruments['result'])

