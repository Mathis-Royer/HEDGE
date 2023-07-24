##import module
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, save_model
from keras.callbacks import ModelCheckpoint

##

def neuronalNetwork(Train, data, new_name_file, old_name_file, epoch, day_windows, nb_forward_candle, plot, accuracyBool):

    ##___________________________________________________
    ##___________________Preprocessing___________________
    X_data = data[['close', 'open', 'high', 'low', 'average_price', 'BearsPower', 'BullsPower', 'Var_DEMA', 'Var_Tenkan', 'Var_SSB', 'Var_SSA', 'MACD', 'RSI']].values
    Y_data = data['close'].values

    scaler = MinMaxScaler(feature_range=(0,1))
    X_scaled_data = scaler.fit_transform(X_data)
    Y_scaled_data = scaler.fit_transform(Y_data.reshape(-1,1))
    ##___________________________________________________
    ##___________________Split dataset___________________
    if not(Train):
        split=0.01
    else:
        split=0.5

    training_data_len = math.ceil(len(X_data)* split)

    X_train_data = X_scaled_data[: training_data_len, :]
    Y_train_data = Y_scaled_data[: training_data_len, :]
    x_train = []
    y_train = np.empty(shape=(nb_forward_candle,len(Y_train_data)-nb_forward_candle-day_windows))

    for i in range(day_windows, len(X_train_data)-nb_forward_candle):
        windows_x=[]
        for j in range(1,day_windows+1):
            windows_x.append(X_train_data[i-j])
        x_train.append(windows_x)

        for j in range(nb_forward_candle):
            y_train[j][i-day_windows] = Y_train_data[i+j]

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], -1, 1))
    y_train = np.reshape(y_train, (y_train.shape[0], -1, 1))

    #-------

    Y_test_data = Y_scaled_data[training_data_len: , : ]
    X_test_data = X_scaled_data[training_data_len: , : ]
    y_test = np.empty(shape=(nb_forward_candle,len(Y_test_data)-nb_forward_candle-day_windows))
    x_test = []

    for i in range(day_windows, len(X_test_data)-nb_forward_candle):
        windows_x=[]
        for j in range(1,day_windows+1):
            windows_x.append(X_test_data[i-j])
        x_test.append(windows_x)

        for j in range(nb_forward_candle):
            y_test[j][i-day_windows] = Y_test_data[i+j]

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], -1, 1))
    y_test = np.reshape(y_test, (y_test.shape[0], -1, 1))
    ##___________________________________________________
    ##_________________Model architecture________________
    if Train and (old_name_file == None or old_name_file == " ") :

        tf.random.set_seed(7) # fix random seed for reproducibility

        input_layer = layers.Input(shape=(x_train.shape[1],1))
        hidden_layers1 = layers.LSTM(200, return_sequences=True)(input_layer)
        hidden_layers1 = layers.LSTM(125, return_sequences=False)(hidden_layers1)
        hidden_layers1 = layers.Dense(100)(hidden_layers1)
        hidden_layers1 = layers.Dense(75)(hidden_layers1)
        hidden_layers1 = layers.Dense(75)(hidden_layers1)
        hidden_layers1 = layers.Dense(50)(hidden_layers1)
        hidden_layers1 = layers.Dense(50)(hidden_layers1)
        hidden_layers1 = layers.Dense(50)(hidden_layers1)
        hidden_layers1 = layers.Dense(25)(hidden_layers1)
        hidden_layers1 = layers.Dense(25)(hidden_layers1)
        outputs=[]
        outputs.append(layers.Dense(x_train.shape[1])(hidden_layers1))


        model = keras.Model(inputs=input_layer, outputs=outputs)

        model.summary()
        model.compile(optimizer='adam', loss={'dense_8':'mse', 'dense_17':'mse','dense_26':'mse', 'dense_35':'mse','dense_44':'mse'})
    ##___________________________________________________
    ##_____________________Train, Test___________________
    print("Fiting model and data procedure :\n")

    #EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.003, patience=2, verbose=1, mode='auto', baseline=0.1)

    if old_name_file != None and old_name_file != " " :
        old_filepath = f'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Include/Hedge_include/saved_best_models/MultiOutput/{old_name_file}'
        model = load_model(old_filepath)

    if Train:
        new_filepath = f'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Include/Hedge_include/saved_best_models/MultiOutput/{new_name_file}'
        model.fit(x_train, y={'dense_8':y_train[0],'dense_17':y_train[1],'dense_26':y_train[2],'dense_35':y_train[3],'dense_44':y_train[4]}, batch_size= 1, epochs=epoch, verbose=1)
        #save_model(model,new_filepath)


    predictions_test = model.predict(x_test)
    for i in range(nb_forward_candle):
        predictions_test[i] = scaler.inverse_transform(predictions_test[i])

    print(y_test.shape)
    print(len(predictions_test[0]))

    predictions_train = model.predict(x_train)
    for i in range(nb_forward_candle):
        predictions_train[i] = scaler.inverse_transform(predictions_train[i])
    ##___________________________________________________
    ##_________________Save or not model_________________
    """
    filepath = 'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Include/Hedge_include/saved_best_models/EURUSD_2LSTM_5DENSE_multiOutput.h5'

    save_model(model,filepath)

    best_model = load_model(filepath)
    loss_best_model = best_model.evaluate(x_test, y_test, verbose=1)
    loss = model.evaluate(x_test, y_test, verbose=0)

    if loss < loss_best_model :
        model.save(filepath)
    """
    ##___________________________________________________
    ##______________________Metrics______________________

    for i in range(nb_forward_candle):
        y_test[i] = scaler.inverse_transform(y_test[i])

    for i in range(nb_forward_candle):
        y_train[i] = scaler.inverse_transform(y_train[i])

    def pourcentageAccuracy(y_test,y_pred):
        accuracy=0

        for i in range(len(y_test)):
            accuracy+=100*math.exp(min(y_pred[i][0],y_test[i][0])/max(y_pred[i][0],y_test[i][0])-1)**1000

        accuracy/=len(y_test)

        return accuracy

    accuracy_test = []
    for i in range(nb_forward_candle):
        accuracy_test.append(pourcentageAccuracy(y_test[i],predictions_test[i]))

    accuracy_train=[]
    for i in range(nb_forward_candle):
        accuracy_train.append(pourcentageAccuracy(y_train[i],predictions_train[i]))

    print("accuracy test = ", accuracy_test)
    print("accuracy train = ", accuracy_train)

    #-------
    rmse_test=[]
    for i in range(nb_forward_candle):
        rmse_test.append(np.sqrt(np.mean(predictions_test[i] - y_test[i])**2))

    print("rmse test = ", rmse_test)

    rmse_train=[]
    for i in range(nb_forward_candle):
        rmse_train.append(np.sqrt(np.mean(predictions_train[i] - y_train[i])**2))

    print("rmse train = ", rmse_train)
    ##___________________________________________________
    ##_________________Print Predictions_________________
    if plot:

        close = data.filter(['close'])
        train = close[:training_data_len-day_windows-nb_forward_candle]
        test = close[training_data_len+day_windows:len(close)-nb_forward_candle]

        for i in range(nb_forward_candle):
            train[f'Predictions{i}'] = predictions_train[i]

        for i in range(nb_forward_candle):
            test[f'Predictions{i}'] = predictions_test[i]

        plt.figure(figsize=(12,6))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')

        plt.plot(train['close'])
        plt.plot(test['close'])

        for i in range(nb_forward_candle):
            plt.plot(train[f'Predictions{i}'])

        for i in range(nb_forward_candle):
            plt.plot(test[f'Predictions{i}'])

        plt.legend(['train', 'test'], loc='lower right')

        #------- 4 plot
        """
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(test[['close','Predictions0']])
        axs[0, 0].set_title('Predictions i+1')
        axs[0, 1].plot(test[['close','Predictions1']])
        axs[0, 1].set_title('Predictions i+2')
        axs[1, 0].plot(test[['close','Predictions2']])
        axs[1, 0].set_title('Predictions i+3')
        axs[1, 1].plot(test[['close','Predictions3']])
        axs[1, 1].set_title('Predictions i+4')

        for ax in axs.flat:
            ax.set(xlabel='Date', ylabel='Close Price USD ($)')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        """
        #-------

        plt.grid()
        plt.show()

    if accuracyBool:
        return accuracy_test
    return rmse_test

##________________________________________________________________________________________________
##________________________________________________________________________________________________
##||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##________________________________________________________________________________________________
##________________________________________________________________________________________________

#0
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-1.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures0 = neuronalNetwork(Train=True,data=data, new_name_file="EURUSD_2j-RFECV_1-TEST_Output-0",old_name_file=None, epoch=1, day_windows=2, nb_forward_candle=5, plot=True, accuracyBool=True)

#1
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-1.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures1 = neuronalNetwork(Train=True,data=data, new_name_file="EURUSD_2j-RFECV_1-1",old_name_file=None, epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#2
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-2.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures2 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-2",old_name_file="EURUSD_2j-RFECV_1-1", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#3
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-3.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures3 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-3",old_name_file="EURUSD_2j-RFECV_1-2", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#4
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-4.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures4 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-4",old_name_file="EURUSD_2j-RFECV_1-3", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#5
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-5.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures5 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-5",old_name_file="EURUSD_2j-RFECV_1-4", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#6
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-6.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures6 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-6",old_name_file="EURUSD_2j-RFECV_1-5", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#7
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-7.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures7 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-7",old_name_file="EURUSD_2j-RFECV_1-6", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#8
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-8.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures8 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-8",old_name_file="EURUSD_2j-RFECV_1-7", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#9
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-9.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures9 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-9",old_name_file="EURUSD_2j-RFECV_1-8", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#10
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-10.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures10 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-10",old_name_file="EURUSD_2j-RFECV_1-9", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#11
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-11.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures11 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-11",old_name_file="EURUSD_2j-RFECV_1-10", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#12
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-12.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures12 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-12",old_name_file="EURUSD_2j-RFECV_1-11", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#13
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-13.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures13 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-13",old_name_file="EURUSD_2j-RFECV_1-12", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#14
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-14.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures14 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-14",old_name_file="EURUSD_2j-RFECV_1-13", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#15
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-15.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeatures15 = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-RFECV_1-15",old_name_file="EURUSD_2j-RFECV_1-14", epoch=3, day_windows=5, nb_forward_candle=5, plot=False, accuracyBool=True)

#Test
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-test.csv',encoding = "ISO-8859-1",sep='\t')

metrics_allFeaturesTest = neuronalNetwork(Train=False, data=data, new_name_file="EURUSD_2j-RFECV_1-test",old_name_file="EURUSD_2j-RFECV_1-15", epoch=3, day_windows=5, nb_forward_candle=5, plot=True, accuracyBool=True)

"""
print("metrics_allFeatures = ", metrics_allFeatures11,metrics_allFeatures12,metrics_allFeatures13,metrics_allFeatures14,metrics_allFeatures15,metrics_allFeatures_test)
"""

