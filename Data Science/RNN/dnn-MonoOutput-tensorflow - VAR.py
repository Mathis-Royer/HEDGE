##import module
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# from sklearn.model_selection import RandomizedSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import pickle
import os
os.chdir("C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Include/Hedge_include")
from RFECV import RFECV_RandomForest


##
def neuronalNetwork(Train, data, new_name_file, old_name_file, max_iter, day_windows, plot_features_importance, plot_loss_curve, plot_roc, plot, grid_search, RFECV_step, split):
    ##___________________________________________________
    ##___________________Preprocessing___________________
    features = ['close', 'open', 'high', 'low', 'average_price', 'BearsPower', 'BullsPower',  'DEMA', 'Var_DEMA', 'Tenkan', 'Kijun','Var_Tenkan', 'Var_SSB', 'Var_SSA', 'MACD', 'RSI']
    X_data = data[features].values
    Y_data = data['close'].values

    scaler = StandardScaler()
    X_scaled_data = scaler.fit_transform(X_data)
    ##___________________________________________________
    ##___________________Split dataset___________________
    if not(Train):
        split=0.01
    else:
        split=0.8

    training_data_len = math.ceil(len(X_data)* split)

    X_train_data = X_scaled_data[: training_data_len, :]
    Y_train_data = Y_data[: training_data_len]
    x_train = []
    y_train = []

    for k in range(day_windows, len(X_train_data)-5):
        #x_train
        windows_x=[]
        for j in range(1,day_windows+1):
            windows_x.append(X_train_data[k-j])
        x_train.append(windows_x)

        #y_train
        for j in range(1,6):
            var=0
            if abs(Y_train_data[k]-Y_train_data[k+j])>data['spread'][k]:
                if Y_train_data[k]-Y_train_data[k+j]<0 and (Y_train_data[k]-Y_train_data[k+1])*(Y_train_data[k]-Y_train_data[k+j])>0:
                    var=1
                    break
                elif Y_train_data[k]-Y_train_data[k+j]>0 and (Y_train_data[k]-Y_train_data[k+1])*(Y_train_data[k]-Y_train_data[k+j])>0:
                    var=2
                    break
        y_train.append(var)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))

    #-------

    Y_test_data = Y_data[training_data_len:]
    X_test_data = X_scaled_data[training_data_len:, :]
    y_test = []
    x_test = []

    for k in range(day_windows, len(X_test_data)-5):
        #x_test
        windows_x=[]
        for j in range(1,day_windows+1):
            windows_x.append(X_test_data[k-j])
        x_test.append(windows_x)

        #y_test
        for j in range(1,6):
            var=0
            if abs(Y_test_data[k]-Y_test_data[k+j])>data['spread'][k]:
                if Y_test_data[k]-Y_test_data[k+j]<0 and (Y_test_data[k]-Y_test_data[k+1])*(Y_test_data[k]-Y_test_data[k+j])>0:
                    var=1
                    break
                elif Y_test_data[k]-Y_test_data[k+j]>0 and (Y_test_data[k]-Y_test_data[k+1])*(Y_test_data[k]-Y_test_data[k+j])>0:
                    var=2
                    break
        y_test.append(var)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    ##___________________________________________________
    ##_________________Features Selection________________
    rfecv=[]
    if RFECV_step:
        print("RFECV procedure :\n")
        rfecv = RFECV_RandomForest(x_test[:int(len(x_test)*split)], y_test[:int(len(y_test)*split)], RFECV_step)
    ##___________________________________________________
    ##_________________Model architecture________________
    if Train and (old_name_file == None or old_name_file == " "):

        if grid_search :
            from sklearn.model_selection import GridSearchCV

            param_grid = {
                'hidden_layer_sizes' : [(len(features),400,300,200,150,125,100,75,50,25,1),(len(features),300,200,150,125,100,75,50,25,1),(len(features),200,150,125,100,75,50,25,1),(len(features),150,125,100,75,50,25,1),(len(features),100,75,50,25,1),(len(features),100,50,1)],
                'max_iter': [50, 75, 100, 150],
                'activation': ['tanh', 'relu', 'logisitic'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant','adaptive','invscaling'],
                'batch_size' : [25,50,100,'auto'],
                'random_state' : [1],
                'verbose' : [True]
            }
            grid = GridSearchCV(MLPClassifier(), param_grid, n_jobs= -1, cv=5)
            grid.fit(x_train, y_train)
            print(grid.best_params_)

        else :
            model = MLPClassifier(solver='adam', activation='relu', learning_rate='invscaling', alpha=0.0001, batch_size= 'auto', max_iter=max_iter, hidden_layer_sizes=(len(features),400,300,200,150,125,100,75,50,25,1), random_state=1, verbose=True)

        ##
        """
        if grid_search:
            from sklearn.model_selection import ParameterGrid

            grid = {'n_estimators': [68,69,70,71], 'max_depth': [32,33,34,35,36], 'max_features': ['sqrt'], 'random_state': [20], 'min_samples_split':[2], 'min_samples_leaf':[1]}
            test_scores = []

            model = RandomForestClassifier()

            for g in ParameterGrid(grid):
                model.set_params(**g)
                model.fit(x_train, y_train)
                test_scores.append(model.score(x_test, y_test))

            best_index = np.argmax(test_scores)
            print(test_scores[best_index], ParameterGrid(grid)[best_index])

        else :
            model = RandomForestClassifier(criterion = 'gini', max_features='sqrt', n_estimators=500, random_state=20, min_samples_split=2, min_samples_leaf=1, max_depth=100, bootstrap=True)
        """
    ##___________________________________________________
    ##______________Train, Test & save Model_____________
    print("Fiting model and data procedure :\n")

    if old_name_file != None and old_name_file != " " :
        old_filepath = f'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Include/Hedge_include/saved_best_models/MonoOutput/{old_name_file}'
        model = pickle.load(open(old_filepath, 'rb'))

    if Train:
        new_filepath = f'C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/24F345EB9F291441AFE537834F9D8A19/MQL5/Include/Hedge_include/saved_best_models/MonoOutput/{new_name_file}'
        model.fit(x_train,y_train)
        pickle.dump(model, open(new_filepath, 'wb'))
        #print(model.out_activation_)

    if plot_features_importance:
        importances = model.feature_importances_
        sorted_index = np.argsort(importances)[::-1]
        x_values = range(len(importances))
        labels = np.array(features)[sorted_index]
        plt.bar(x_values, importances[sorted_index], tick_label=labels)
        plt.xticks(rotation=90)
        plt.show()

    if plot_loss_curve :
        plt.plot(model.loss_curve_)
        plt.title("Loss Curve", fontsize=14)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.show()

    predictions_test=model.predict(x_test)
    predictions_train=model.predict(x_train)

    if plot_roc:
        """from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve

        train_roc_auc = roc_auc_score(y_train, predictions_train, labels=[0,1,2])
        test_roc_auc = roc_auc_score(y_test, predictions_test, labels=[0,1,2])

        train_false_pred, train_true_pred, thresholds = roc_curve(y_train, predictions_train[:,1])
        test_false_pred, test_true_pred, thresholds = roc_curve(y_test, predictions_test[:,1])

        plt.plot(train_false_pred, train_true_pred, label='Train (area = %0.2f)' % train_roc_auc)
        plt.plot(test_false_pred, test_true_pred, label='Test (area = %0.2f)' % test_roc_auc)

        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic ROC')
        plt.legend(loc="lower right")

        plt.show()"""
    ##___________________________________________________
    ##______________________Metrics______________________

    def pourcentageAccuracy(y_test,y_pred):
        faux_positifs=0
        faux_negatifs=0
        vrai=0
        vrai_neutre=0
        buy_reel=0
        buy_predit=0
        sell_reel=0
        sell_predit=0

        for i in range(len(y_test)):
            if y_test[i]==1:
                buy_reel+=1
            if y_test[i]==2:
                sell_reel+=1

            if y_pred[i]==2 and y_test[i]!=2:
                faux_negatifs+=1
            elif y_pred[i]==1 and y_test[i]!=1:
                faux_positifs+=1

            elif y_pred[i]==1 and y_test[i]==1:
                buy_predit+=1
            elif y_pred[i]==2 and y_test[i]==2:
                sell_predit+=1
            elif y_pred[i]==0 and y_test[i]==0:
                vrai_neutre+=1

        print("buy_reel =",buy_reel)
        print("sell_reel =",sell_reel)

        vrai = buy_predit + sell_predit
        nb_erreur = faux_positifs + faux_negatifs
        nb_total_order = nb_erreur + vrai

        accuracy_buy = 100*buy_predit/nb_total_order
        accuracy_sell = 100*sell_predit/nb_total_order

        print("\nFaux buy = %.2f%%"%(100*faux_positifs/nb_total_order),"\t\t-\tFaux sell = %.2f%%"%(100*faux_negatifs/nb_total_order))

        print("Vrai buy = %.2f%%"%accuracy_buy,"\t\t-\tVrai sell = %.2f%%"%accuracy_sell)

        return (100*vrai/nb_total_order,100*nb_erreur/nb_total_order)

    accuracy_test = pourcentageAccuracy(y_test,predictions_test)
    print("accuracy test = %.2f%%"%accuracy_test[0], "\t-\terreurs = %.2f%%"%accuracy_test[1])
    accuracy_train = pourcentageAccuracy(y_train,predictions_train)
    print("accuracy train = %.2f%%"%accuracy_train[0],"-\terreurs = %.2f%%"%accuracy_train[1])
    ##___________________________________________________
    ##_________________Print Predictions_________________
    if plot:

        buy=np.zeros(len(data)-day_windows)
        buy[:] = np.nan
        sell=np.zeros(len(data)-day_windows)
        sell[:] = np.nan

        for k in range(len(y_train)):
            if predictions_train[k]==1:
                buy[k]=data['close'][day_windows+k]
            elif predictions_train[k]==2:
                sell[k]=data['close'][day_windows+k]
        for k in range(len(y_test)):
            if predictions_test[k]==1:
                buy[len(y_train)+k]=data['close'][day_windows+len(y_train)+k]
            elif predictions_test[k]==2:
                sell[len(y_train)+k]=data['close'][day_windows+len(y_train)+k]

        buy=pd.DataFrame(buy)
        sell=pd.DataFrame(sell)

        plt.figure(figsize=(12,6))
        plt.title('Model')
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')

        plt.plot(data['close'].values[day_windows:], color='grey')
        plt.scatter(buy.index,buy,color='green', lw=0.05)
        plt.scatter(sell.index,sell,color='red',lw=0.05)
        plt.legend(['Close Price', 'Predictions Buy', 'Predictions Sell'], loc='lower right')

        plt.grid()
        plt.show()

    return accuracy_test, rfecv

##________________________________________________________________________________________________
##________________________________________________________________________________________________
##||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
##________________________________________________________________________________________________
##________________________________________________________________________________________________

#1
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_5j-1.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures1, rfecv = neuronalNetwork(Train=True,data=data, new_name_file="EURUSD_2j-Var_SkL-1",old_name_file=None, max_iter=200, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)
"""
#2
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-2.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures2, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-2",old_name_file="EURUSD_2j-Var_SkL-1", max_iter=2000, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#3
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-3.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures3, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-3",old_name_file="EURUSD_2j-Var_SkL-2", max_iter=2000, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#4
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-4.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures4, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-4",old_name_file="EURUSD_2j-Var_SkL-3", max_iter=2000, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#5
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-5.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures5, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-5",old_name_file="EURUSD_2j-Var_SkL-4", max_iter=2000, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#6
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-6.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures6, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-6",old_name_file="EURUSD_2j-Var_SkL-5", max_iter=2000, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#7
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-7.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures7, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-7",old_name_file="EURUSD_2j-Var_SkL-6", max_iter=2000, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#8
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-8.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures8, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-8",old_name_file="EURUSD_2j-Var_SkL-7", max_iter=200, day_windows=15, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#9
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-9.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures9, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-9",old_name_file="EURUSD_2j-Var_SkL-8", max_iter=200, day_windows=5, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#10
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-10.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures10, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-10",old_name_file="EURUSD_2j-Var_SkL-9", max_iter=200, day_windows=5, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#11
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-11.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures11, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-11",old_name_file="EURUSD_2j-Var_SkL-10", max_iter=200, day_windows=5, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#12
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-12.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures12, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-12",old_name_file="EURUSD_2j-Var_SkL-11", max_iter=200, day_windows=5, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#13
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-13.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures13, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-13",old_name_file="EURUSD_2j-Var_SkL-12", max_iter=200, day_windows=5, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#14
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-14.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures14, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-14",old_name_file="EURUSD_2j-Var_SkL-13", max_iter=200, day_windows=5, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#15
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-15.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures15, rfecv = neuronalNetwork(Train=True, data=data, new_name_file="EURUSD_2j-Var_SkL-15",old_name_file="EURUSD_2j-Var_SkL-14", max_iter=200, day_windows=5, plot_features_importance=False, plot_loss_curve=True, plot_roc=False, plot=False, grid_search=False, RFECV_step=0, split=0.1)

#Test
data = pd.read_csv('C:/Users/royer/AppData/Roaming/MetaQuotes/Terminal/Common/Files/EURUSD_2j-test.csv',encoding = "ISO-8859-1",sep='\t')
data = data[::-1].reset_index()
data.pop('index')

metrics_allFeatures_test, rfecv = neuronalNetwork(Train=False, data=data, new_name_file=None,old_name_file="EURUSD_2j-Var_SkL-10", max_iter=200, day_windows=5, plot_features_importance=True, plot_loss_curve=True, plot_roc=True, plot=True, grid_search=False, RFECV_step=0, split=1)
"""