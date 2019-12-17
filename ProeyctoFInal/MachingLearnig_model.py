import numpy 
import numpy as np
import matplotlib.pyplot as plt
import pandas 
from keras.models import Sequential
from keras.layers import Dense
import math
import pandas as pd
import scipy
# fix random seed for reproducibility
numpy.random.seed(7)    

datapath = 'airline-passengers.csv'



#dataframe = pandas.read_csv(datapath, usecols=[1], engine='python', skipfooter=3)
#print("----- ",dataframe)

df = pd.read_csv("1.csv", sep=";", header=0)
datos_workload = df["workload"]

array = np.zeros((len(datos_workload),1))

for x in range(len(datos_workload)):
    array[x][0]=datos_workload[x];
    
print(array[0:5],array.shape)

dataset = array
#dataset = dataset.astype('float32')
print("+++++++++",array,array.shape)



# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(len(train), len(test))

def create_dataset(dataset, look_back=1,look_forward=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        b = dataset[(i+look_back):(i+look_back+look_forward),0]
        dataX.append(a)
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)

look_back = 4
look_forward =2
trainX, trainY = create_dataset(train, look_back,look_forward)
testX, testY = create_dataset(test, look_back,look_forward)



# create and fit Multilayer Perceptron model
model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(4))
model.add(Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=200, batch_size=2, verbose=2)
#------------------------------------------------------------------------------
# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print("sdasdasdasd",testPredict )

print(trainPredict.shape)
print(trainPredict[:,0])

trainPredict1 =trainPredict[:,0]
trainPredict11 = trainPredict1.reshape(55,1)
print(trainPredict11.shape)

trainPredict2 =trainPredict[:,1]
trainPredict12 = trainPredict2.reshape(55,1)
print(trainPredict12.shape)

# shift train predictions for plotting
trainPredictPlot1 = numpy.empty_like(dataset)
trainPredictPlot1[:, :] = numpy.nan
trainPredictPlot1[look_back:len(trainPredict)+look_back, :] = trainPredict11


testPredict1 =testPredict[:,0]
print("sdasdasdasd",testPredict1 )
testPredict11 = testPredict1.reshape(25,1)
####################################################

####################################################



# shift test predictions for plotting
testPredictPlot1 = numpy.empty_like(dataset)
testPredictPlot1[:, :] = 1#numpy.nan
testPredictPlot1[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict11


def mean_absolute_percentage_error(y_true, y_pred): 
    
    print(len(y_true), len(y_pred))
    y_pred = [numpy.round(x) for x in y_pred]
    #print(y_true, y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# plot baseline and predictions
plt.figure(figsize=(18,6))
plt.plot(dataset , color='blue',label='Datos reales')
plt.plot(trainPredictPlot1,color="green",label='Datos de entrenamiento')
plt.plot(testPredictPlot1,color="red",label='Datos predecidos')
plt.legend(loc='upper right')
plt.show()



print(mean_absolute_percentage_error(dataset,testPredictPlot1));
