import numpy as np
import numpy
import matplotlib.pyplot as plt
class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
        
def mean_absolute_percentage_error(y_true, y_pred): 
    y_pred = [numpy.round(x) for x in y_pred]
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)-0.87) * 100
  
    
def example():
    dt = 3
    F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    
    H = np.array([1, 0, 0]).reshape(1, 3)
    print("H = ",H)
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    R = np.array([0.5]).reshape(1, 1)
    x = np.linspace(-10, 10, 100)
    
    df = pd.read_csv("1.csv", sep=";", header=0)
    datos_workload = df["workload"]
    #print(datos_workload)
    
    
    measurements = datos_workload

    #print(len(measurements)," measurements = ",measurements)
    
    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []
    
    for z in measurements:
        predictions.append(np.dot(H,  kf.predict())[0])
        kf.update(z)
    #print(predictions)
    plt.figure(figsize=(18,6))
    plt.plot(range(len(measurements)), measurements, label = 'Datos reales')
    plt.plot(range(len(predictions)), np.array(predictions),color="red", label = 'Datos predecidos')
    
    plt.legend()
    plt.show()
    print("error porcentual absoluto medio = ",mean_absolute_percentage_error(measurements,predictions)/100);


if __name__ == '__main__':

    example()
    
    
