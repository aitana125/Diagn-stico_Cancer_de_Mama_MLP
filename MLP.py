import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Funciones Auxiliares

def sigmoide(z):
    return 1 / (np.exp(-z) + 1)

def sigmoide_derivada(a):
    return a*(1-a)

def forward_pass(x, W1, W2, b1, b2):
    z1 = np.dot(x, W1) + b1
    a1 = sigmoide(z1)
   
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoide(z2)
    return a1, a2

def backpropagation(x, Y, a1, a2, W2):
    d2 = (a2 - Y) * sigmoide_derivada(a2)
    dW2 = np.dot(a1.T, d2)
    db2 = np.sum(d2, axis=0)
   
    d1 = np.dot(d2, W2.T) * sigmoide_derivada(a1)
    dW1 = np.dot(x.T, d1)
    db1 = np.sum(d1, axis=0)
   
    return dW1, db1, dW2, db2


#np.random.seed(42)

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


W1 = np.random.randn(30, 15) * 0.1  # número de entradas, número en la capa oculta
b1 = np.random.randn(15) * 0.1
W2 = np.random.randn(15, 1) * 0.1  # número en la capa oculta, número de salidas 
b2 = np.random.randn(1) * 0.1
# incializamos los pesos y bias con valores aleatorios, multiplicando con 0.1 para crear valores pequeños
# y evitar que la función de activación se sature.

num_epoc = 200
ET = np.zeros(num_epoc)  # error total
learning_rate = 0.01  # condición de patada

for epoca in range(num_epoc):
    
    # 2. Forward pass
    a1, a2 = forward_pass(X_train, W1, W2, b1, b2)
   
    # 3. Cálculo del Error
    Matriz_E = 0.5 * (Y_train - a2)**2
    ET[epoca] = np.mean(np.sum(Matriz_E, axis=1))
   
    # 4. Backpropagation
    dW1, db1, dW2, db2 = backpropagation(X_train, Y_train, a1, a2, W2)
   
    # 5. Actualizar pesos
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    
# 6. Prueba de la red ya entrenada con datos de prueba

_, y_modelo = forward_pass(X_test, W1, W2, b1, b2)       # forwar pass con los datos de prueba
pred = (y_modelo > 0.5).astype(int)      # si la Y del modelo es mayor a 0.5, clasificamos como Maligno
precision = np.mean(pred == Y_test) * 100
print(f"Precisión: {precision:.2f}%")


# 7. Crear figura del Error

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epoc + 1), ET, color='blue')

plt.xlabel('Época')
plt.ylabel('Error')
plt.title('Progresión del error')

plt.grid(True)
plt.show()