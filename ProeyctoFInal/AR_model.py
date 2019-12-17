from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AR
from random import random
import numpy as np


def metodo_AR(series,training_percentage):     
    X = series.values
    train_size = int(len(X))
   
    train, test = X[:train_size], X[:train_size]
    print(len(train)," ",len(test))
    
    #print("test = ",len(test))
    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    
    for t in range(len(test)):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	obs = test[t]
    	predictions.append(yhat)
    	history.append(obs)
    	#print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    return test, predictions, error

def mean_absolute_percentage_error(y_true, y_pred): 
    
    
    #print(y_true,y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv("1.csv", sep=";", header=0)
datos_workload = df["workload"]

test, predictions, mse = metodo_AR(datos_workload,training_percentage=.7)
#print("\n\nTest Mean absolute: %.3f; Mean Squared Error: %.3f" , mse,"\n\n")
# plot
plt.figure(facecolor='w')
plt.figure(figsize=(18,6))
pyplot.plot(test,label='Datos reales')
pyplot.plot(predictions, color='red',label='Datos predecidos')
plt.legend(loc='upper right')
pyplot.show()
print("error porcentual absoluto medio = ",mean_absolute_percentage_error(test,predictions)/100);