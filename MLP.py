import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(42)
df = pd.read_csv("dataset_limpio.csv")

# 1. División y Estandarización

muestras = 569
X = df.drop(columns=['diagnosis']).values
Y = df['diagnosis'].values
# Obtenemos únicamente las variables del dataset

# Hacemos una división del 80% 20%
idx = np.random.permutation(muestras) # crea valores aleatorios del 0 al 568
num_m = int(muestras * 0.8)

X_train = X[idx[:num_m]]
Y_train = Y[idx[:num_m]]

X_test = X[idx[num_m:]]
Y_test = Y[idx[num_m:]]

# Normalizamos los valores via estandarización

mean_train = np.mean(X_train, axis=0)
se_train = np.std(X_train, axis=0)

# La media y desviación es calculada solo con el conjunto de entrenamiento

X_train = (X_train - mean_train) / se_train
X_test= (X_test - mean_train) / se_train
# La fórmula de estandarización es (x - media) / desviación estandar (Z-score)

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)
# para que el broadcasting funcione y la resta a la hora de
# hacer el backpropagation no cause ningún error