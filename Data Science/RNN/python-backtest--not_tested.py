import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

## Load & preprocess data

data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-test.csv',encoding = "ISO-8859-1",sep='\t')

# data = data[::-1].reset_index()
# data.pop('index')

X_data = data[['close', 'open', 'high', 'low', 'average_price', 'BearsPower', 'BullsPower', 'Var_DEMA', 'Var_Tenkan', 'Var_SSB', 'Var_SSA', 'MACD', 'RSI']].values

scaler = MinMaxScaler(feature_range=(0,1))
X_scaled_data = scaler.fit_transform(X_data)
scaler.fit(data['close'].values.reshape(-1,1))

x = []
candle_windows = 5

for i in range(candle_windows, len(X_scaled_data)):
    windows_x=[]
    for j in range(1,candle_windows+1):
        windows_x.append(X_scaled_data[i-j])
    x.append(windows_x)

x = np.array(x)
x = np.reshape(x, (x.shape[0], -1, 1))

## Run model

old_filepath = f'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Include/Hedge_include/saved_best_models/MonoOutput/EURUSD_2j-43Features-15'
model = load_model(old_filepath)

predictions = model.predict(x)
predictions = scaler.inverse_transform(predictions)

variation=[]

for i in range(len(predictions)-5):
        variation.append([predictions[i+1][0]-predictions[i][0],predictions[i+2][0]-predictions[i][0],predictions[i+3][0]-predictions[i][0],predictions[i+4][0]-predictions[i][0]])

##

order = np.zeros(len(variation),dtype=int)

already_open=False
last_open=0

print("debut")
for k in range(len(variation)):
    ##CLOSE ORDER
    if(already_open and last_open==1 and variation[k][0]<0 and data['spread'][candle_windows+k]*2<abs(variation[k][0])):
        order[k]=3
        already_open=False
        last_open=0

    elif(already_open and last_open==2 and variation[k][0]>0 and data['spread'][candle_windows+k]*2<abs(variation[k][0])):
        order[k]=3
        already_open=False
        last_open=0

    ##OPEN ORDER
    for j in range(len(variation[0])):

        if(not(already_open) and variation[k][0]*variation[k][j]>0 and variation[k][j]>0 and data['spread'][candle_windows+k]*2<variation[k][j]):
                order[k]+=1
                already_open=True
                last_open=1
                break

        elif(not(already_open) and variation[k][0]*variation[k][j]>0 and variation[k][j]<0 and data['spread'][candle_windows+k]*2<abs(variation[k][j])):
                order[k]+=2
                already_open=True
                last_open=2
                break

## Plot

nb=[k for k in range(len(order))]

order_sell = np.zeros(len(order))
order_buy = np.zeros(len(order))

last_order=0
nb_sell=0
nb_buy=0

for k in range(len(order)):
    if order[k]==5 or order[k]==4:
        order_sell[k]=data['close'].values[10+k]
        nb_sell+=1

        order_buy[k]=data['close'].values[10+k]
        nb_buy+=1

        if order[k]==5:
            last_order=2
        else :
            last_order=1

    elif (order[k]==3 and last_order==1) or order[k]==1:
        order_buy[k]=data['close'].values[10+k]
        last_order=1
        nb_buy+=1

    elif (order[k]==3 and last_order==2) or order[k]==2:
        order_sell[k]=data['close'].values[10+k]
        last_order=2
        nb_sell+=1

    else:
        if nb_sell%2 ==0:
            order_sell[k]=0.0
        else :
            order_sell[k]=np.nan

        if nb_buy%2 ==0:
            order_buy[k]=0.0
        else:
            order_buy[k]=np.nan


order_buy = pd.DataFrame(order_buy)
order_buy = np.reshape(np.array(order_buy.interpolate(limit_area='inside').fillna(0)),(-1))
order_sell = pd.DataFrame(order_sell)
order_sell = np.reshape(np.array(order_sell.interpolate(limit_area='inside').fillna(0)),(-1))

for k in range(len(order_sell)):
    if order_sell[k]==0.0:
        order_sell[k]=np.nan
    if order_buy[k]==0.0:
        order_buy[k]=np.nan


plt.figure()
plt.plot(data['close'].values[candle_windows+5:])
plt.plot([k for k in range(len(data)-10)], predictions[:-5])
plt.plot(nb, order_sell,color='red')
plt.plot(nb,order_buy, color='green')
plt.legend(['price','predictions'])
plt.show()

## Export data

zeros=[0 for k in range(10)]
order = zeros + list(order)

np.savetxt("C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/Orders.csv", order, delimiter =", ", fmt ='%d')





