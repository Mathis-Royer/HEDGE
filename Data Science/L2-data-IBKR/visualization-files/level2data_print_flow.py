#Import
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
import threading
import time
from datetime import datetime
from datetime import timezone
from datetime import timedelta
from blessings import Terminal

#Class for Interactive Brokers Connection
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
    #Listen for realtime bars
    def updateMktDepth(self, reqId, position: int, operation: int, side: int, price: float, size):
        bot.market_depth_update(reqId, position, operation, side, price, size)

#Bot Logic
class Bot():
    ib = None
    
    def __init__(self, Minute):
        self.Minute = Minute
        self.insert_list = []
        
        #Connect to IB on init
        self.ib = IBapi()
        self.ib.connect ('127.0.0.1', 7496, 1)
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        time.sleep(1)
        
        #Create our IB contract object
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType ="CASH"
        contract.exchange = "IDEALPRO"
        contract.currency = "USD"
        
        # Request Market Data
        self.ib.reqMktDepth(0, contract, 20, False, [])
    
    #Listen to socket in seperate thread
    def run_loop(self):
        self.ib.run()

    #Pass realtime bar data back to our bot object
    def market_depth_update(self, reqId, position, operation, side, price, size):
        #operation type
        if operation==0 :
            operation='insert'
        elif operation==1 :
            operation='update'
        else:
            operation='delete'
        
        #pre-processing for printing
        if price*100000-(int(price*10000)*10) == 0:
            price=str(price)+'0'
        else:
            price=str(price)
             
        # if size/100000<1:
        #     size=str(size)+'.0000'
        # elif size/100000<10:
        #     size=str(size)+'.000'
        # elif size/100000<100:
        #     size=str(size)+'.00'
        # elif size/100000<1000:
        #     size=str(size)+'.0'
        # else:
        #     size=str(size)

        #order type
        if side==0:
            side='ask'
        else:
            side='bid'
        
        t = Terminal()

        DateTime = datetime.now(timezone(timedelta(hours=+2)))
        minute = DateTime.strftime("%M")
        

        if position==0:
            if side=='ask':
                print(t.on_red(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_green(f'{side}'),t.on_red(f'PRICE : {price}$  SIZE :{size}'))
            else:
                print(t.on_red(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_red(f'{side}'),t.on_red(f'PRICE : {price}$  SIZE :{size}'))
            
        elif position==1:
            if side=='ask':
                print(t.on_magenta(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_green(f'{side}'),t.on_magenta(f'PRICE : {price}$  SIZE :{size}'))
            else:
                print(t.on_magenta(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_red(f'{side}'),t.on_magenta(f'PRICE : {price}$  SIZE :{size}'))
            
        elif position==2:
            if side=='ask':
                print(t.on_blue(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_green(f'{side}'),t.on_blue(f'PRICE : {price}$  SIZE :{size}'))
            else:
                print(t.on_blue(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_red(f'{side}'),t.on_blue(f'PRICE : {price}$  SIZE :{size}'))
            
        elif position==3:
            if side=='ask':
                print(t.on_green(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_green(f'{side}'),t.on_green(f'PRICE : {price}$  SIZE :{size}'))
            else:
                print(t.on_green(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_red(f'{side}'),t.on_green(f'PRICE : {price}$  SIZE :{size}'))
            
        else:
            if side=='ask':
                print(t.on_yellow(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_green(f'{side}'),t.on_yellow(f'PRICE : {price}$  SIZE :{size}'))
            else:
                print(t.on_yellow(f'POSITION : {position} OPERATION : {operation} SIDE :'),t.bold_red(f'{side}'),t.on_yellow(f'PRICE : {price}$  SIZE :{size}'))
            
            
            
            
            
#Start bot

hour = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%H'))
Minute = int(datetime.now(timezone(timedelta(hours=+2))).strftime('%M'))
day = datetime.now(timezone(timedelta(hours=+2))).strftime('%a')

if True :#day != 'Sat' and day != 'Sun' and day != 'Fri':
    bot = Bot(Minute)
elif day=='Fri' and hour<22:
    bot = Bot(Minute)
elif day=='Fri' and hour==22 and Minute<55:
    bot = Bot(Minute)