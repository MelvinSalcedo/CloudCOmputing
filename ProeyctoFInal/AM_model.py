# MA example
import numpy as np
from statsmodels.tsa.arima_model import ARMA
from random import random
import matplotlib.pyplot as plt
# contrived dataset
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

df = pd.read_csv("1.csv", sep=";", header=0)
datos_workload = df["workload"]

data =datos_workload# [x + random() for x in range(1, 100)]

#print(data)
# fit model
model = ARMA(data, order=(0, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(0,len(data)-1)
#print("yat = ",yhat)


plt.figure(facecolor='w')
plt.figure(figsize=(18,6))
pyplot.plot(data,label='Datos reales')
pyplot.plot(yhat, color='red',label='Datos predecidos')
plt.legend(loc='upper right')
pyplot.show()
print(mean_absolute_percentage_error(data,yhat));