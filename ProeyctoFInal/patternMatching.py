from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
#pattern matching algorithm performs 2 steps: Preprocess and Match. It returns the predicated list after performing the match
'''
KMP implementation
    # a is bigger
    # b is smaller
'''
def KMP(str_a, str_b):
    return _KMP(str_a,str_b,0)

def _KMP(str_a, str_b, offset_a):
    """
    :type offset_a: int # returns the starting index of matching text(smaller) in pattern(bigger)
    """
    if (len(str_a) - offset_a < len(str_b)):
        return -1
    for i in range(offset_a, offset_a + len(str_b), 1):
        if (str_b[i - offset_a] != str_a[i]):
            return _KMP(str_a, str_b, i+1)
    return offset_a

'''
Calculating euclidean distance of a character from a string character
'''
def euclidean_distance(str, ch):
    sum=0
    #print ("Euclidean")
    #print (str)
    #print (ch)
    n = len(str)
    for index in range(0, len(str)):
        difference = ord(str[index])-ord(ch)
        sum += difference ** 2
    sum = sum ** (0.5)
    return sum

def pattern_matching(cpu_workload):
    '''
    :param cpu_workload: list
    :return: string
     Two steps:
        1. Preprocessing
        2. Match
    '''
    # In this algorithm we get take string x, Observed CPU Workload Time series and after calculations get String z.
    # cpu_workload = [3, 15, 16, 18, 19, 15, 20, 15, 15, 12, 14, 16, 19, 14, 14, 22, 16, 16, 13, 15]

    #print("X = ",cpu_workload)
    #cpu_workload = [3, 15, 18]
    Y = []
    i = 0
    for index in range(len(cpu_workload) - 1):
        y = cpu_workload[index + 1] - cpu_workload[index]
        Y.insert(i, y)
        i += 1
    #print("Y = ",Y)
    z = []
    
    l = 0
    # building z
    for index in range(len(Y)):
        if (Y[index] == 0):
            z.insert(l, 'i')
        elif (Y[index] == 1):
            z.insert(l, 'k')
        elif (Y[index] == 2 or Y[index] == 3):
            z.insert(l, 'l')
        elif (Y[index] == 4 or Y[index] == 5):
            z.insert(l, 'm')
        elif (int(Y[index]) > 5):
            z.insert(l, 'n')
        elif (Y[index] == -1 or Y[index] == -2):
            z.insert(l, 'g')
        elif (Y[index] == -3 or Y[index] == -4):
            z.insert(l, 'f')
        elif (Y[index] <= -5):
            z.insert(l, 'e')
        l += 1

    # s[1..l] is the output of pre-processing step
    #print("Z = ",z)
    # End of preprocessing step
    # s and C_his_pattern_string are input to 2nd step


    #Match Step
    C_his_pattern_list = [['e', 'f', 'g'], ['i', 'k', 'l'], ['m', 'n', 'f']]
    num = len(C_his_pattern_list) - 1

    # ing = ''.join(s)
    C_his_pattern_string = ''
    for x in C_his_pattern_list:
        for y in x:
            C_his_pattern_string += y

    # converting to strings
    #print("C = ",C_his_pattern_string)

    # Initializing dis[] with zeroes
    dis = []
    for index in range(num + 1):
        dis.insert(i, 0)

    s_size = len(z) - 1
    
    while (num != 0):
        length = len(C_his_pattern_list[num])
        # s is smaller string, C_his_pattern_list[num-1] is bigger one
        tag = KMP(C_his_pattern_list[num - 1], z)
        #print("tag = ",tag)

        if (tag != -1):
            return (C_his_pattern_list[num])

        else:
            dis[num] = float('inf')
            for i in range(0, length - s_size):
                dis_l = euclidean_distance(z, C_his_pattern_list[num][i])
                #print("Dis_l")
                #print(dis_l)
                if dis_l < dis[num]:
                    dis[num] = dis_l
                    
        num -= 1
        
    
    min = dis[0]
    min_index = 0
    for index in range(len(dis)):
        #print(index)
        if dis[index] < min:
            min = dis[index]
            min_index = index

    #print(min)
    #print(min_index)
    #print(C_his_pattern_list[num])
    return C_his_pattern_list[num]


def metodo_Dm(cpu_workload,Y,Z,output_list):     
    
    X = Y
    train_size = int(len(X))
   
    train, test = X[:train_size], X[:train_size]
    #print(len(train)," ",len(test))
    
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
    
    print(len(Z),"  ",len(output_list)," ",  len(test))
    
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1]
        obs = test[t]
        if (Z[t]==output_list[0] ):#or Z[t]==output_list[1] or Z[t]==output_list[2]):
            predictions.append(cpu_workload[t])
        else:
            predictions.append(-yhat+4)
        history.append(obs)
    	#print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    return test, predictions, error

def predecir(cpu_workload,output_list):
    Y = []
    z = []
    
    i = 0
    for index in range(len(cpu_workload) - 1):
        y = cpu_workload[index + 1] - cpu_workload[index]
        Y.insert(i, y)
        i += 1
    #print("Y = ",Y)
    Y.insert(i, 1)
    l = 0
    for index in range(len(Y)):
        if (Y[index] == 0):
            z.insert(l, 'i')
        elif (Y[index] == 1):
            z.insert(l, 'k')
        elif (Y[index] == 2 or Y[index] == 3):
            z.insert(l, 'l')
        elif (Y[index] == 4 or Y[index] == 5):
            z.insert(l, 'm')
        elif (int(Y[index]) > 5):
            z.insert(l, 'n')
        elif (Y[index] == -1 or Y[index] == -2):
            z.insert(l, 'g')
        elif (Y[index] == -3 or Y[index] == -4):
            z.insert(l, 'f')
        elif (Y[index] <= -5):
            z.insert(l, 'e')
        l += 1
        
    print(l," y= ",len(Y)," z= ",len(z),"  ",len(output_list)," x=",  len(cpu_workload))
    test, predictions, mse = metodo_Dm(cpu_workload,Y,z,output_list)
    return predictions;

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

if __name__ == "__main__":

   output_list = []
   #cpu_workload = [13.5, 15.3, 18.9]
   df = pd.read_csv("1.csv", sep=";", header=0)
   datos_workload = df["workload"]
   
   cpu_workload = datos_workload#[3, 15, 16, 18, 19, 15, 20, 15, 15, 12, 14, 16, 19, 14, 14, 22, 16, 16, 13, 15]
   #results = cpu_util.cpu_utilizations()
   #print(results)
   
   output_list = pattern_matching(cpu_workload)
   print("*********************");
   print ("C[num] = ",output_list)
   print("*********************");
   
   Predict_workload =predecir(cpu_workload,output_list)
   
   plt.figure(facecolor='w')
   plt.figure(figsize=(18,6))
   pyplot.plot(cpu_workload,label='Datos reales')
   pyplot.plot(Predict_workload, color='red',label='Datos predecidos')
   plt.legend(loc='upper right')
   pyplot.show()

print(mean_absolute_percentage_error(datos_workload,Predict_workload));