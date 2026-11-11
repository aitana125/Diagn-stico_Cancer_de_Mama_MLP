import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

# 1. Limpieza de datos

print(df.columns)

"""
Primero identificamos el nombre de las columnas para decidir cuales quitar por completo

['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
'fractal_dimension_se', 'radius_worst', 'texture_worst',
'perimeter_worst', 'area_worst', 'smoothness_worst',
'compactness_worst', 'concavity_worst', 'concave points_worst',
'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32']

"""

df = df.drop(columns = ['id', 'Unnamed: 32']) 
df.index.name = "Num_Muestra"
# El dataset contiene una columna de 'id' que identifica la masa, y la columna 'unnamed:32' está llena de nulos
# Nombramos el index que pandas asigna automáticamente al dataset

nulos = df.isnull().sum()
print("Columnas con valores nulos:\n\n", nulos[nulos>0])
# En este caso, no hay ninguna columna con valores faltantes


# 2. Creación de Diagramas

# Creamos unos diagramas que exploren la naturaleza de las variables para decidir
# si es necesario normalizar las variables, y en tal caso que tipo de normalización.

# 1. Distribución de tumores con un Bar Chart

fig0, ax0 = plt.subplots(figsize=(5, 5))

sns.countplot(data=df, x='diagnosis', palette=['#ff9999','#66b3ff'])
ax0.set_title('Tumores Benignos vs Malignos)')


# 2. Figura de Histogramas y Scatter Plots

fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))
fig1.suptitle("Distribuciones y Relaciones según el Diagnóstico")

# Histograma Radio Medio
sns.histplot(data=df, x='radius_mean', hue='diagnosis', kde=True, ax=axes1[0, 0], palette=['#ff9999','#66b3ff'])
axes1[0, 0].set_title('Distribución del Radio Medio')

# Histograma Concavidad Media
sns.histplot(data=df, x='concavity_mean', hue='diagnosis', kde=True, ax=axes1[0, 1], palette=['#ff9999','#66b3ff'])
axes1[0, 1].set_title('Distribución de la Concavidad Media')

# Scatter Plot Radio medio vs Peor Textura
sns.scatterplot(data=df, x='radius_mean', y='texture_worst', hue='diagnosis', ax=axes1[1, 0], palette=['#ff9999','#66b3ff'])
axes1[1, 0].set_title('Relación Radio vs Peor Textura')

# Scatter Plot Radio medio vs Peor Textura
sns.scatterplot(data=df, x='smoothness_mean', y='smoothness_se', hue='diagnosis', ax=axes1[1, 1], palette=['#ff9999','#66b3ff'])
axes1[1, 1].set_title('Relación Radio vs Peor Textura')


# 3. Figura de Boxplots

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))
fig2.suptitle("Medias según el Diagnóstico")

# Área media
sns.boxplot(data=df, x='diagnosis', y='area_mean', ax=axes2[0, 0], palette=['#ff9999','#66b3ff'])
axes2[0, 0].set_title("Área Media")

# Radio medio
sns.boxplot(data=df, x='diagnosis', y='radius_mean', ax=axes2[0, 1], palette=['#ff9999','#66b3ff'])
axes2[0, 1].set_title("Radio Medio")

# Concavidad media
sns.boxplot(data=df, x='diagnosis', y='concavity_mean', ax=axes2[1, 0], palette=['#ff9999','#66b3ff'])
axes2[1, 0].set_title("Concavidad Media")

# Perimetro medio
sns.boxplot(data=df, x='diagnosis', y='perimeter_mean', ax=axes2[1, 1], palette=['#ff9999','#66b3ff'])
axes2[1, 1].set_title("Perimetro Medio")


plt.show()

# Como hemos observado, hay muchas variables que tienen outliers
# por lo que una normalización con la función (min-max) / max
# no es la mejor opción en este caso. Por ello vamos a estandarizar.

# Para evitar una fuga de datos, la estandarización se hará en
# el archivo MLP donde se encontrará todo el código que entrenará la
# red neuronal.


# 3. Variable de Salida a Binario

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# df.to_csv("dataset_limpio.csv", index=False)
# para crear el nuevo dataset con los cambios realizados