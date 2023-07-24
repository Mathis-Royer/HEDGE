#Import
from ibapi.client import EClient
from multiprocessing import active_children
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
from datetime import datetime
from datetime import timezone
from datetime import timedelta
import csv
import os
import sys
import pandas as pd
import numpy as np
import decimal
from statistics import median
from L2_functions import create_VP_template_file, extend_VP_headers, add_a_new_row

#________________________________________________________________________________________________

#Class for Interactive Brokers Connection
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
    #Listen for realtime bars
    def updateMktDepth(self, reqId, position: int, operation: int, side: int, price: float, size):
        bot.market_depth_update(reqId, position, operation, side, price, size)

#________________________________________________________________________________________________

#Bot Logic
class Bot():
    ib = None
    
    def __init__(self, Minute, Hour):
        symbols = ['EURUSD','EURGBP','GBPCHF','GBPUSD','USDCHF','USDJPY','EURJPY','EURCHF','GBPJPY','USDCAD','AUDUSD','NZDUSD','CADJPY','NZDJPY','AUDCAD','EURCAD','AUDNZD','CHFJPY','NZDCAD','GBPCAD','GBPNZD','AUDCHF','NZDCHF','CADCHF','EURCNH','AUDJPY','USDCNH','GBPAUD','EURAUD','EURNZD']
        self.ATH_and_ATL = []
        df = pd.read_csv('symbols_ATL_ATH_accuracy.csv', index_col=False)
        
        #Connect to IB on init
        self.ib = IBapi()
                
        begin=0
        end=3
        children=[]
        
        for k in range(begin,end):
            pid_k = os.fork()
            if pid_k:
                time.sleep(1)
                children.append(pid_k)
            else:
                hour = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%H'))
                day = datetime.now(timezone(timedelta(hours=+2))).strftime('%a')    
                while (day != 'Sat' and day != 'Sun' and day != 'Fri') or (day=='Fri' and hour<23) or (day=='Sun' and hour>=23):
                    hour = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%H'))
                    day = datetime.now(timezone(timedelta(hours=+2))).strftime('%a')
                    
                    pid = os.fork()
                    if pid:
                        os.waitpid(pid, 0)
                        print(f'\n----  ALERT : STOP EXECUTION : {symbols[k]}')
                        time.sleep(1)
                    else:
                        print(f'\n----  STARTING : {symbols[k]}')
                        
                        self.ib.connect('127.0.0.1', 7496, k)
                        ib_thread = threading.Thread(target=self.run_loop)
                        ib_thread.start()
                                                
                        ATL_symbole = df.loc[df['symbol']==symbols[k],'ATL'].values[0]
                        ATH_symbole = df.loc[df['symbol']==symbols[k],'ATH'].values[0]
                        step = decimal.Decimal(df.loc[df['symbol']==symbols[k],'accuracy'].values[0][1:-1])
                        self.ATH_and_ATL.append(ATH_symbole)
                        self.ATH_and_ATL.append(ATL_symbole)
                        self.ATH_and_ATL.append(step)
                        
                        self.symbol = symbols[k]
                        self.begin = True
                        self.Minute = Minute
                        self.first_bid=0
                        self.high = 0
                        self.low = 0
                        self.last_bid = 0
                        self.last_ask = 0
                        self.spread=None
                        self.volume_profil_hour = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%H'))
                        self.ask_volume = []
                        self.bid_volume = []
                        self.ask_price = []
                        self.bid_price = []
                        self.VP_price = []
                        self.VP_volume = []

                        contract = Contract()
                        contract.symbol = symbols[k][:3]
                        contract.secType ="CASH"
                        contract.exchange = "IDEALPRO"
                        contract.currency = symbols[k][3:]
                        self.ib.reqMktDepth(k, contract, 20, False, [])
                        break
                break
        if pid_k:
            for k in range(end-begin):
                os.waitpid(children[k],0)
                print(f'----  ALERT : STOP EXECUTION PARENT OF : {symbols[k]}\n')
        
#________________________________________________________________________________________________

    #Listen to socket in seperate thread
    def run_loop(self):
        self.ib.run()

#________________________________________________________________________________________________

    #Pass realtime bar data back to our bot object
    def market_depth_update(self, reqId, position, operation, side, price, size):
        DateTime = datetime.now(timezone(timedelta(hours=+2)))
        minute = int(DateTime.strftime("%M"))
        hour = int(DateTime.strftime("%H"))
        day = int(DateTime.strftime("%d"))
        if ((day != 'Sat' and day != 'Sun' and day != 'Fri') or (day=='Fri' and hour<23) or (day=='Sun' and hour>=23)):
            if price != 0 :
                
                #begin
                if self.begin:
                    self.first_bid=price
                    self.high=price
                    self.low=price
                    self.Minute=minute
                    print(f'----  START : {self.symbol} : {minute}\n')
                    self.begin = False
                
                
                #select data
                if side==0:
                    self.last_ask=price
                    
                    #calcul ask price volume  
                    if float(price) in self.ask_price:
                        self.ask_volume[self.ask_price.index(price)].append(size)
                    else:
                        self.ask_volume.append([size])
                        self.ask_price.append(price)

                    #save higher and lower price
                    if float(price) > self.high:
                        self.high = float(price)
                    elif float(price) < self.low:
                        self.low = float(price)
                
                else:
                    self.last_bid=price
                    
                    #calcul Volume Profile volume
                    if position==0:
                        if float(price) in self.VP_price:
                            self.VP_volume[self.VP_price.index(price)]+=1
                        elif price !=0 :
                            self.VP_price.append(price)
                            self.VP_volume.append(1)
                    
                    #calcul ask price volume
                    if float(price) in self.bid_price:
                        self.bid_volume[self.bid_price.index(price)].append(size)
                    elif price != 0:
                        self.bid_volume.append([size])
                        self.bid_price.append(price)
                
                #export pre-process data
                if minute != self.Minute:
                    print(f'----  NEW MINUTE : {self.symbol} : {minute}\n')
                    
                    DateTime = datetime.now(timezone(timedelta(hours=+2)))        
                    month = int(DateTime.strftime("%m"))
                    day = int(DateTime.strftime("%d"))
                    year = int(DateTime.strftime("%Y"))
                    hour = int(DateTime.strftime("%H"))
                    
                    #median price
                    ask_median = round(decimal.Decimal(median(self.ask_price))/self.ATH_and_ATL[2])*self.ATH_and_ATL[2]
                    bid_median = round(decimal.Decimal(median(self.bid_price))/self.ATH_and_ATL[2])*self.ATH_and_ATL[2]
                    
                    #spread
                    last_spread=self.ATH_and_ATL[2]
                    if self.spread != None:
                        last_spread = self.spread
                    self.spread = round(abs(decimal.Decimal(self.last_bid)-decimal.Decimal(self.last_ask))/self.ATH_and_ATL[2])
                    if self.spread==0:
                        self.spread = last_spread
                    
                    #high & low processing
                    self.high = round((decimal.Decimal(self.high)-decimal.Decimal(self.last_bid))/decimal.Decimal(self.ATH_and_ATL[2]))
                    self.low = round((decimal.Decimal(self.last_ask)-decimal.Decimal(self.low))/decimal.Decimal(self.ATH_and_ATL[2]))
                    
                    #pre-processing
                    ask_volume_T=[]
                    for volumes in self.ask_volume:
                        sommes=0
                        nb=0
                        for k in range(len(volumes)):
                            sommes+=volumes[k]
                            nb=k
                        if nb !=0:
                            ask_volume_T.append(sommes/nb)
                        else:
                            ask_volume_T.append(sommes)
                    
                    ask_price_T=[]
                    for prices in self.ask_price:
                        ask_price_T.append(round((decimal.Decimal(self.last_ask)-decimal.Decimal(prices))/decimal.Decimal(self.ATH_and_ATL[2])))
                    
                    bid_volume_T=[]
                    for volumes in self.bid_volume:
                        sommes=0
                        nb=0
                        for k in range(len(volumes)):
                            sommes+=volumes[k]
                            nb=k
                        if nb !=0:
                            bid_volume_T.append(sommes/nb)
                        else:
                            bid_volume_T.append(sommes)
                    
                    bid_price_T=[]
                    for prices in self.bid_price:
                        bid_price_T.append(round((decimal.Decimal(self.last_bid)-decimal.Decimal(prices))/decimal.Decimal(self.ATH_and_ATL[2])))
                    
                    #Volume cumulé
                    if ask_volume_T==[]:
                        ask_volume_total = 0
                    else:
                        ask_volume_total = round(sum(ask_volume_T))
                    if bid_volume_T==[]:
                        bid_volume_total=0
                    else:
                        bid_volume_total = round(sum(bid_volume_T))
                    
                    #Volume profil
                    if int(hour) != int(self.volume_profil_hour):
                        print(f'----  NEW HOUR : {self.symbol} : {self.volume_profil_hour} --> {hour}')
                        ATL,ATH=self.ATH_and_ATL[1],self.ATH_and_ATL[0]
                        
                        file_exists = os.path.isfile(f'L2/VP_{self.symbol}.csv')
                        #if file doesn't exist
                        if not file_exists:
                            create_VP_template_file(self.symbol, ATL,ATH,self.ATH_and_ATL[2])
                        
                        
                        if min(self.bid_price) < ATL and min(self.bid_price)!=0 :   #If one of the recorded prices is lower than the previous historical minimum
                            ATL = min(self.bid_price)
                        if max(self.ask_price) > ATH:   #If one of the recorded prices is higher than the previous historical maximum
                            ATH = max(self.ask_price)
                            
                        
                        #if new extreme value, so new header
                        if file_exists and (ATL != self.ATH_and_ATL[1] or ATH != self.ATH_and_ATL[0]):                
                            extend_VP_headers(self.symbol,ATL,ATH,self.ATH_and_ATL[2],self.ATH_and_ATL[1],self.ATH_and_ATL[0])
                                
                        
                        #bid volume profile processing and export
                        VP_min_price = str(min(self.VP_price))
                        if len(VP_min_price)!=len(str(self.ATH_and_ATL[2])):
                            for k in range(len(str(self.ATH_and_ATL[2]))-len(VP_min_price)):
                                VP_min_price+='0'
                        VP_max_price = str(max(self.VP_price))
                        if len(VP_max_price)!=len(str(self.ATH_and_ATL[2])):
                            for k in range(len(str(self.ATH_and_ATL[2]))-len(VP_max_price)):
                                VP_max_price+='0'
                                
                        VP_volume_T = []
                        range_insert = int((decimal.Decimal(VP_max_price) - decimal.Decimal(VP_min_price))/self.ATH_and_ATL[2])
                        
                        for k in range(range_insert):
                            k_price = float(decimal.Decimal(VP_min_price) + self.ATH_and_ATL[2]*k)
                            if k_price in self.VP_price:
                                VP_volume_T.append(self.VP_volume[self.VP_price.index(k_price)])
                            else:
                                VP_volume_T.append(0)
                        
                        df = pd.read_csv(f'L2/VP_{self.symbol}.csv')
                        df.iloc[len(df.index)-1,0:4] = [int(year),int(month),int(day),int(hour)]
                        df.iloc[len(df.index)-1,df.columns.get_loc(VP_min_price):df.columns.get_loc(VP_min_price)+range_insert] = VP_volume_T
                        df.to_csv(f'L2/VP_{self.symbol}.csv', index=False)
                        add_a_new_row(f'L2/VP_{self.symbol}.csv')
                        
                        self.VP_price = []
                        self.VP_volume = []
                        self.volume_profil_hour = hour
                        
                    
                    #limit bid_price_T, ask_price_T, bid_volume_T, ask_volume_T to 10 data
                    pd_bid_volume_T = pd.Series(bid_volume_T, dtype=float)
                    id = pd_bid_volume_T.nlargest(10)
                    id_lst = id.index.values.tolist()
                    bid_volume_T = id.values.tolist()
                    bid_volume_T = [round(bid_volume_T[k]) for k in range(len(bid_volume_T))]
                    bid_price_T = [bid_price_T[k] for k in id_lst]
                    if len(bid_price_T)<10:
                        for k in range(10-len(bid_price_T)):
                            bid_price_T.append(0)
                            bid_volume_T.append(0)
                    
                    pd_ask_volume_T = pd.Series(ask_volume_T, dtype=float)
                    id = pd_ask_volume_T.nlargest(10)
                    id_lst = id.index.values.tolist()
                    ask_volume_T = id.values.tolist()
                    ask_volume_T = [round(ask_volume_T[k]) for k in range(len(ask_volume_T))]
                    ask_price_T = [ask_price_T[k] for k in id_lst]
                    if len(ask_price_T)<10:
                        for k in range(10-len(ask_price_T)):
                            ask_price_T.append(0)
                            ask_volume_T.append(0)
                    
                    file_exists = os.path.isfile(f'L2/{self.symbol}_{year}_{month}_{day}.csv')
                    
                    #export
                    with open(f'L2/{self.symbol}_{year}_{month}_{day}.csv', 'a') as file:
                        headers = ['hour', 'minute', 'tendance', 'last_ask', 'last_bid', 'first_bid', 'candle_body', 'high', 'low', 'spread','ask_median','bid_median', 'ask_volume_total', 'bid_volume_total', 'volume_total', 'bid_price1', 'bid_price2', 'bid_price3', 'bid_price4', 'bid_price5', 'bid_price6', 'bid_price7', 'bid_price8', 'bid_price9', 'bid_price10', 'bid_volume1', 'bid_volume2', 'bid_volume3', 'bid_volume4', 'bid_volume5', 'bid_volume6', 'bid_volume7', 'bid_volume8', 'bid_volume9', 'bid_volume10', 'ask_price1', 'ask_price2', 'ask_price3', 'ask_price4', 'ask_price5', 'ask_price6', 'ask_price7', 'ask_price8', 'ask_price9', 'ask_price10', 'ask_volume1', 'ask_volume2', 'ask_volume3', 'ask_volume4', 'ask_volume5', 'ask_volume6', 'ask_volume7', 'ask_volume8', 'ask_volume9', 'ask_volume10']
                        writer = csv.DictWriter(file,fieldnames=headers)
                        
                        if not file_exists:
                            writer.writeheader()  # file doesn't exist yet, write a header
                    
                    df = pd.read_csv(f'L2/{self.symbol}_{year}_{month}_{day}.csv')
                    
                    if self.last_bid - self.first_bid > 0:
                        tendance=1
                    else :
                        tendance=0
                    
                    data=[hour]
                    data.append(minute)
                    data.append(tendance)
                    data.append(self.last_ask)
                    data.append(self.last_bid)
                    data.append(self.first_bid)
                    data.append(round((decimal.Decimal(self.last_bid)-decimal.Decimal(self.first_bid))/decimal.Decimal(self.ATH_and_ATL[2])))
                    data.append(self.high)
                    data.append(self.low)
                    data.append(self.spread)
                    data.append(ask_median)
                    data.append(bid_median)
                    data.append(ask_volume_total*100/(ask_volume_total+bid_volume_total))
                    data.append(bid_volume_total*100/(ask_volume_total+bid_volume_total))
                    data.append(ask_volume_total+bid_volume_total)
                    data+=bid_price_T
                    data+=bid_volume_T
                    data+=ask_price_T
                    data+=ask_volume_T
                    
                    df.loc[len(df.index)] = data
                    df.to_csv(f'L2/{self.symbol}_{year}_{month}_{day}.csv', index=None)
                    
                    #reinitialise list
                    self.ask_volume = []
                    self.ask_price = []
                    self.bid_volume = []
                    self.bid_price = []
                    self.Minute = minute
                    
                    self.first_bid=self.last_bid
                    self.high = self.last_bid
                    self.low = self.last_bid
        else:
            return
                
             
#________________________________________________________________________________________________
#________________________________________________________________________________________________
#________________________________________________________________________________________________
#Start bot

hour = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%H'))
day = datetime.now(timezone(timedelta(hours=+2))).strftime('%a')

restart = 0
while (day != 'Sat' and day != 'Sun' and day != 'Fri') or (day=='Fri' and hour<23) or (day=='Sun' and hour>=23):
    hour = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%H'))
    day = datetime.now(timezone(timedelta(hours=+2))).strftime('%a')
    
    children = 0
    pid = os.fork()
    if pid:
        done = os.waitpid(pid, 0)
        print(f'Perte de connection n°{restart} - done :', done)
        time.sleep(1)
    else:
        Minute = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%M'))
        print('----------------------------------------------------------------------------')
        print(f'<<<<<-------  Start main program n°{restart} : {day} {hour}:{Minute}  ------->>>>>')
        print('----------------------------------------------------------------------------')

        bot = Bot(Minute, hour)
        break
    restart+=1
if pid:
    print("<<<<<-------  Bourses Fermées  ------->>>>>")