#import necessary modules
import csv
import pandas as pd

df = pd.read_csv("1.csv", sep=";", header=0)
datos_workload = df["workload"]

print(datos_workload)