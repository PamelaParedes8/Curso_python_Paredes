#Trabajo-Final
#Autora: Pamela Paredes

### Variable clave asignada: categoria_cocina
### Población objetivo: region == "Costa"

# ***********IMPORTACIÓN DE LIBRERIAS ******************
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ****************** EJERCICIO 1 ************************

### Importación y limpieza de la base de datos
### Importación de la base de datos de la Encuesta Nacional de Desnutrición Infantil (ENDI) del Ecuador:
datos = pd.read_table("Data/sample_endi_model_10p.txt", sep=";")
print(datos)
# La base de datos tiene información sobre 2237 niños (2237 observaciones).

### Variables (columnas) de la base de datos:
datos.columns

#Convirtiendo etiquetas numericas a etiqueta más compensibles para la variable región:
datos["region"] = datos["region"].apply(lambda x: "Costa" if x == 1 else "Sierra" if x == 2 else "Oriente")

### Filtración de la base según la población objetivo:
filtro= datos["region"] == "Costa" 
datos_1= datos[filtro] # Utiliza un booleano (True/False) como filtro y solo se queda con los casos que cumplen la condicion
print(datos_1)
# Existen 1075 observaciones, es decir, la base de datos contiene informaicón sobre 1075 de niños que viven en la región Costa

### Eliminando casos (filas) con valores faltantes de la variable dependiente dummy (columna) dcronica:
datos_1 = datos_1[~datos_1["dcronica"].isna()] 
print(datos_1)
# La base de datos depurada contiene información sobre 1055 niños.

### Seleccionando variables claves para el análisis:
variables = ['n_hijos', 'region', 'sexo','condicion_empleo','categoria_cocina']

### Elimina valores faltantes de cada una de esas variables:
for i in variables:
    datos_1 = datos_1[~datos_1[i].isna()]

print (datos_1) #Dataframe depurado con 1055 observaciones
#Interpretación: Se analizarán 1055 niños de la región Costa.

###*************CÁLCULO DE LAS ESTADÍSTICAS BÁSICAS ****************

## Frecuencia: manera en que cocina los alimentos el hogar del niño:
frecuencias= datos_1.groupby("categoria_cocina").size()
print(frecuencias)
# Interpretación: En la región Costa, en 1016 hogares de los niños se cocina con gas/electricidad, en 38 hogares se cocinan con leña o carbón y en 1 hogar no se cocina.

## Moda de la variable "manera de cocinar los alimentos (categoria_cocina)" para los niños de la región Costa:

moda= datos_1["categoria_cocina"].mode()
print (moda) #Calcula la moda y no la media ya que categoria_cocina es una variable cualitativa 
#Interpretación: la manera de cocinar los alimentos en los hogares de los niños que predomina en la muestra es gas/electricidad

# ******************* EJERCICIO 2: MODELO LOGIT *****************
### MODELO DE VALIDACIÓN CRUZADA

## Transformación de variables
# Definiendo las variables cuantitativas y cualitativas (categóricas) a utilizar:

# Definimos las variables categóricas y numéricas que utilizaremos en nuestro análisis
variables_categoricas = ['region', 'sexo', 'condicion_empleo', 'categoria_cocina']
variables_numericas = ['n_hijos']

# Transformador para estandarizar las variables numéricas y una copia de los datos:
transformador = StandardScaler()
datos_escalados = datos_1.copy()
datos_escalados[variables_numericas] = transformador.fit_transform(datos_escalados[variables_numericas])

# Convirtiendo las variables cualitativas categóricas en dummies:
datos_dummies = pd.get_dummies(datos_escalados, columns=variables_categoricas, drop_first=True)
#datos_dummies.columns
# Seleccionando la variable dependiente (Y) e independientes (X) para el modelo:

x = datos_dummies[['n_hijos', 'sexo_Mujer', 
                   'condicion_empleo_Empleada', 'condicion_empleo_Inactiva', 'condicion_empleo_Menor a 15 años', 'categoria_cocina_Leña o carbón', 'categoria_cocina_No cocina']]
y = datos_dummies["dcronica"] 
#En estas dummies no incluye las categorias que son las categorias BASE. Para la variable de desempleo la categoria base es Desempleada. La variable región ya no la considero porque la base ya está filtrada solo para la región Costa
#La categoría base de la variable categoria_cocina es Gas/Electricidad, por tanto incluyo las otras dummies que no son base.

# Utilizando el factor de expansión para considerar los pesos asociados a cada observación:
weights = datos_dummies['fexp_nino']

## Separación de muestras de entrenamiento (train) y de prueba (test)
#Dividiendo los datos, 80% datos para entrenamiento y 20% datos para prueba:
x_train, x_test, y_train, y_test, weights_train, weights_test = train_test_split(x, y, weights, test_size=0.2, random_state=42)

#Transformando todas las variables a numéricas enteras para poder utilizar un modelo logit:

# Convirtiendo todas las variables a tipo numérico:
x_train = x_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

# Conviertiendo las variables a tipo entero:
variables = x_train.columns
for i in variables:
    x_train[i] = x_train[i].astype(int)
    x_test[i] = x_test[i].astype(int)

y_train = y_train.astype(int)
# Ajustando los datos a un modelo logit:
modelo = sm.Logit(y_train, x_train)
result = modelo.fit()
print(result.summary())

#*******************RESPUESTAS A LAS PREEGUNTAS*********

### ¿Cuál es el valor del parámetro asociado a la variable clave si ejecutamos el modelo solo con el conjunto de entrenamiento y predecimos con el mismo conjunto de entrenamiento? ¿Es significativo?
#El valor del parámetro de la variable categoria_cocina (cuya categoría base es Gas/Electricidad) es 1.5411. 
#Observación: En el modelo se incluyó la categoría "No cocina", sin embargo, dicho valor no se visualiza en el resumen del ajuste del modelo,esto puede deberse a dos posibles razones: 
# 1.Que el conjunto de datos de entrenamiento tiene muy pocas observaciones (hay un solo hogar del niño donde no se cocina) o 2. Ese único hogar donde no se cocina puede ser una observación que se encuentra en el conjunto de testeo (prueba) y no al de entrenamiento con el que se está trabajando.
#Dicho valor del parámetro es significativo al 1%, 5% y 10% ya que es P>|z|=0.000, es decir, la probabilidad es inferior a 0.01, 0.05 y 0.10 por lo que se rechaza la Ho.

### Interpretación de resultados
# El valor del coeficiente  1.5411 no se lo interpreta de manera directa dado que es un modelo logit, por lo que se necesitan calcular los efectos marginales, sin embargo interpretando la significancia y el signo es posible decir que:
# Significancia: La manera de cocinar los alimentos en los hogares de los niños de la región Costa si determina si un niño tiene destrución crónica infantil o no.
# Signo: En la región Costa, la probabilidad de que un niño tenga desnutrición aumenta cuando en el hogar en el que el infante vive se cocina los alimentos con leña o carbón en relación a un hogar en el que se cocina con gas/electricidad, manteniéndose las demás variables constantes.

# ************ EJERCICIO 3: EVALUACIÓN DEL MODELO CON DATOS FITLRADOS ***********
### Extracción de los coeficientes y los almacenamos en un DataFrame
coeficientes = result.params
df_coeficientes = pd.DataFrame(coeficientes).reset_index()
df_coeficientes.columns = ['Variable', 'Coeficiente']

### Creación de una tabla pivote para una mejor visualización
df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
df_pivot.reset_index(drop=True, inplace=True)

### Predicciones en el conjunto de prueba y evaluar su rendimiento:
predictions = result.predict() 

### Conversión de  las probabilidades en clases binarias
predictions_class = (predictions > 0.5).astype(int)

### Comparamos las predicciones con los valores reales
predictions_class == y_test
print("La precisión promedio del modelo testeando con datos test es", np.mean(predictions_class))
 
 ### Validación cruzada
# 100 folds:
kf = KFold(n_splits=100)
accuracy_scores = []
df_params = pd.DataFrame()

for train_index, test_index in kf.split(x_train):

    # aleatorizamos los folds en las partes necesarias:
    x_train_fold, x_test_fold = x_train.iloc[train_index], x_train.iloc[test_index]
    y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
    weights_train_fold, weights_test_fold = weights_train.iloc[train_index], weights_train.iloc[test_index]
    
    # Ajustamos un modelo de regresión logística en el pliegue de entrenamiento
    log_reg = sm.Logit(y_train_fold, x_train_fold)
    result_reg = log_reg.fit()
    
    # Extraemos los coeficientes y los organizamos en un DataFrame
    coeficientes = result_reg.params
    df_coeficientes = pd.DataFrame(coeficientes).reset_index()
    df_coeficientes.columns = ['Variable', 'Coeficiente']
    df_pivot = df_coeficientes.pivot_table(columns='Variable', values='Coeficiente')
    df_pivot.reset_index(drop=True, inplace=True)
    
    # Realizamos predicciones en el pliegue de prueba
    predictions = result_reg.predict(x_test_fold)
    predictions = (predictions >= 0.5).astype(int)
    
    # Calculamos la precisión del modelo en el pliegue de prueba
    accuracy = accuracy_score(y_test_fold, predictions)
    accuracy_scores.append(accuracy)
    
    # Concatenamos los coeficientes estimados en cada pliegue en un DataFrame
    df_params = pd.concat([df_params, df_pivot], ignore_index=True)

print(f"Precisión promedio de validación cruzada: {np.mean(accuracy_scores)}")

### Validación cruzada: precisión del modelo:
# Calculando la precisión promedio
precision_promedio = np.mean(accuracy_scores)
print(precision_promedio)

plt.hist(accuracy_scores, bins=30, edgecolor='black')
# Añadir una línea vertical en la precisión promedio
plt.axvline(precision_promedio, color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la precisión promedio
plt.text(precision_promedio-0.1, plt.ylim()[1]-0.1, f'Precisión promedio: {precision_promedio:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

###Validación cruzada: el comportamiento del parámetro asocaido a n_hijos
plt.hist(df_params["n_hijos"], bins=30, edgecolor='black')

# Añadir una línea vertical en la media de los coeficientes
plt.axvline(np.mean(df_params["n_hijos"]), color='red', linestyle='dashed', linewidth=2)

# Añadir un texto que muestre la media de los coeficientes
plt.text(np.mean(df_params["n_hijos"])-0.1, plt.ylim()[1]-0.1, f'Media de los coeficientes: {np.mean(df_params["n_hijos"]):.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title('Histograma de Beta (N Hijos)')
plt.xlabel('Valor del parámetro')
plt.ylabel('Frecuencia')

# Ajustar los márgenes
plt.tight_layout()

# Mostrar el histograma
plt.show()

#**************** RESPUESTA A LAS PREGUNTAS **************************
##1. ¿Qué sucede con la precisión promedio del modelo cuando se utiliza el conjunto de datos filtrado? (Incremento o disminuye ¿Cuanto?)
# Precisión modelo filtrado: -0.05950
# Precisión modelo sin filtrar: 0.0426540
# Por tanto, la precisión del modelo cuando se utiliza un conjunto de datos filtrados disminuye en 0.1021.   
                                                                                        
##2.¿Qué sucede con la distribución de los coeficientes beta en comparación con el ejercicio anterior? (Incrementa o disminuye ¿Cuanto?)
#La media de los coeficientes de la base filtrada (que incluye la variable de análisis) disminuye en relación al conjunto de datos inicial.