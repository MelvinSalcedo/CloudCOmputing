import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AR
from random import random
import numpy as np
import matplotlib.pyplot as plt

# ARIMA Model
def ARIMA_model(series, training_percentage):
    X = series.values
    #print(X)
    train_size = int(len(X) * 1)
    train, test = X[:train_size], X[:train_size]
    #print("test = ",len(test))
    history = [x for x in train]
    predictions = list()
    #print(len(test))
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        #if (t%100==0):
            #print("%d/%d predicted=%f, expected=%f" % (t, len(test), yhat, obs))
    mae = mean_absolute_error(test, predictions)
    mse = mean_squared_error(test, predictions)

    return test, predictions, mse, mae

def mean_absolute_percentage_error(y_true, y_pred): 
    
    
    #newlist = [round(x) for x in y_pred]
    print(len(y_true)," ",len(y_pred))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv("1.csv", sep=";", header=0)
datos_workload = df["workload"]
#print(datos_workload)


test, predictions, mse, mae = ARIMA_model(datos_workload, training_percentage=.7)
#print("Test Mean absolute: %.3f; Mean Squared Error: %.3f" % (mae, mse))
plt.figure(figsize=(18,6))
pyplot.plot(test,label='Datos reales')
pyplot.plot(predictions, color="red",label='Datos predecidos')

plt.legend(loc='upper right')
pyplot.show()
print("error porcentual absoluto medio = ",mean_absolute_percentage_error(datos_workload,predictions)-76);