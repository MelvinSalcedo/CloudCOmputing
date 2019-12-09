from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AR
from random import random


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def metodo_Dm(series,training_percentage):     
    X = series
    
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
    	predictions.append(-yhat+3)
    	history.append(obs)
    	#print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    return test, predictions, error

df = pd.read_csv("1.csv", sep=";", header=0)
datos_workload = df["workload"]
Y = []
i = 0
for index in range(len(datos_workload) - 1):
    y = datos_workload[index + 1] - datos_workload[index]
    Y.insert(i, y)
    i += 1
Y.insert(i, 1)
test, predictions, mse = metodo_Dm(Y,training_percentage=.7)
#print("\n\nTest Mean absolute: %.3f; Mean Squared Error: %.3f" , mse,"\n\n")
# plot
plt.figure(facecolor='w')
plt.figure(figsize=(18,6))
pyplot.plot(datos_workload,label='Datos reales')
pyplot.plot(predictions, color='red',label='Datos predecidos')
plt.legend(loc='upper right')
pyplot.show()
print(mean_absolute_percentage_error(datos_workload,predictions));