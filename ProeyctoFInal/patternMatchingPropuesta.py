import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
import numpy as np


def Distance(PatternElement, PatternScale, DataElement, DataScale):
    #print("***",PatternElement, PatternScale, DataElement, DataScale)

    a=ord(PatternElement)
    b=ord(PatternScale)
    c=ord(DataElement)
    d=ord(DataScale)
    
    t=a*b-c*d
    #print("t = ",t)
    return t
    

def CumulativeDistance(P, T, DataOffset):
    patternScale = P[0]
    dataScale = T[DataOffset]
    length = len(P)
    distance = 0
    for index in range (length):
        distance = distance + abs( ord(dataScale) * ord(P[index])
        - ord(patternScale) * ord(T[index + DataOffset]) )
    return distance

def Calculate_prefix_approx(P, ACCEPT_INST_ERR):
    phi = np.zeros(len(P))
    
    m = len(P)
    phi[0] = -1
    k = -1
    scaleK = P[0]
    scaleQ = P[1]
    for q in range(1, (m-1)):
        dist = Distance(P[k+1], scaleK, P[q], scaleQ)
        maxDistance = ACCEPT_INST_ERR * ord(scaleQ) * ord(P[k+1])
        while (k > -1 and dist > maxDistance):
            k = phi[k]
            dist = Distance(P[k+1], scaleK, P[q], scaleQ)
            scaleQ = P[q - (k+1)]
        if (dist <= ACCEPT_INST_ERR * ord(scaleQ) * ord(P[k+1])):
            k = k+1
        phi[q] = k
    #print("phi = ",phi)
    return phi


def KMP_approx(T, P, ACCEPT_INST_ERR, ACCEPT_CUMUL_ERR):
    print("------> ",T, P, ACCEPT_INST_ERR, ACCEPT_CUMUL_ERR )
    patternSum=1
    StoreSolution=1
    phi = set()
    n = len(T)
    m = len(P)
    phi = Calculate_prefix_approx(P,ACCEPT_INST_ERR)
    q = -1
    scaleP = P[0]
    scaleT = T[0]
    for i in range(0,n - 1):
        dist = Distance(P[int(q+1)], scaleP, T[i], scaleT)
        maxDist = ACCEPT_INST_ERR * ord(scaleT) * ord(P[int(q+1)])
        while (q > -1 and dist > maxDist):
            dist = Distance(P[q+1], scaleP, T[i], scaleT)
            q = phi[q]
            scaleT = T[i - (q+1)]
            maxDist = ACCEPT_INST_ERR * scaleT * P[q+1]

        if (dist <= maxDist):
            q = q+1

        if (q == m-1):
            dist = CumulativeDistance(P, T, i - m + 1)
            maxDist = ACCEPT_CUMUL_ERR * patternSum * ord(scaleT)
            if (dist <= maxDist):
                StoreSolution=(dist / ord(scaleT), i - m + 1)
            q = int(phi[q])
            suma=int(q+1)
            scaleP = P[suma]
            scaleT = T[i - suma]
    print(q,n,m,phi,n-m)
    return n-m
    if n==m and T[m-1] == P[i-1]:
        return (q+1) - m
    else:
        return -1

"""
def get_prefix_table(patron):
    prefix_set = set()
    m = len(patron)
    prefix_table = [0]*m
    delimeter = 1
    while(delimeter<m):
        prefix_set.add(patron[:delimeter])
        j = 1
        while(j<delimeter+1):
            if patron[j:delimeter+1] in prefix_set:
                prefix_table[delimeter] = delimeter - j + 1
                break
            j += 1
        delimeter += 1
        
    print(prefix_table)
    return prefix_table

def strstr(texto, patron):
    # m: denoting the position within S where the prospective match for W begins
    # i: denoting the index of the currently considered character in W.
    texto_len = len(texto)
    patron_len = len(patron)
    if (patron_len > texto_len) or (not texto_len) or (not patron_len):
        return -1
    prefix_table = get_prefix_table(patron)
    m = i = 0
    while((i<patron_len) and (m<texto_len)):
        if texto[m] == patron[i]:
            i += 1
            m += 1
        else:
            if i != 0:
                i = prefix_table[i-1]
            else:
                m += 1
    if i==patron_len and texto[m-1] == patron[i-1]:
        return m - patron_len
    else:
        return -1
"""
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
        print("++++++++",num)
        length = len(C_his_pattern_list[num])
        # s is smaller string, C_his_pattern_list[num-1] is bigger one
        #tag = KMP(C_his_pattern_list[num - 1], z)
        tag = KMP_approx(z,C_his_pattern_list[num - 1],1,1)
        print("tag= ",tag)
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
            predictions.append(-yhat+3.5)
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
    y_pred= [round(x) for x in y_pred]
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

print("error porcentual absoluto medio = ",mean_absolute_percentage_error(datos_workload,Predict_workload)/100);