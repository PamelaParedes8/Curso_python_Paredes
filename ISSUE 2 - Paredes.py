# ISSUE 2 - Reproducción de lo aprendido en la clase 2

### TEMA: Información Ambiental de Empresas Ecuatorianas del año 2021
import numpy as  np
## Creación de cadenas (variables de texto):

Produccion_alternativa ="Producción de energía eléctrica alternativa generada (kWh/año)"
Consumo_alternativa = "Consumo de energía eléctrica alternativa generada (kWh/año) "
Venta_alternativa = "Venta de energía eléctrica generada (kWh/año) "

print (Produccion_alternativa)
print (Consumo_alternativa) 
print (Venta_alternativa)

## Creación de variables numéricas y lista  lista de números:

# Vector de números flotantes: Valores de producción, consumo y venta de tres empresas durante el año 2021 (kWh):
Produccion = [433123.2, 345673.5, 198452.6]
Consumo = [15426.7, 23461.3, 34243.7]
Venta = [417696.5,322112.2 , 164208.9]
print (Produccion)
print (Consumo)
print(Venta)

# Vector de números enteros: Número de personas que trabajaron en actividades ambientales durante el año 2021 ( tres empresas):
Personal_ambiental= [321,421,129]

## Creación de un diccionario:
#Diccionario del tipo de fuente para generación eléctrica: 1. Solar, 2. Eólica, 3.Biomasa, 4.Hidráulica, 5. Generador termoeléctrico, 6.Otro
Diccionario = { "Solar":"1","Eólica":"2","Biomasa":"3","Hidráulica":"4", "Termoeléctrico":"5", "Otro":"6"}
print (Diccionario)

## Creación de variable lógica
# Respuesta a la pregunta: la empresa generó energía eléctrica alternativa en el año 2021?
Alternativa = [True,False]
print (Alternativa)

##Importación de datos de un archivo de Excel
import pandas as pd
Datos_solar = pd.read_excel("Data/Energia_Solar.xlsx")
print (Datos_solar)
##Cargando esos datos en un dataframe:
df_solar = pd.DataFrame(Datos_solar)
print (df_solar)

##Análisis de tipo de datos que contiene el dataframe:
#Tipos de variables del dataframe
df_solar.dtypes

#Nombre de las variables del dataframe
df_solar.columns

#Número de filas y columnas del dataframe
df_solar.shape

#Estadística descriptiva rápida de las variables cuantitativas del dataframe
df_solar.describe()

#Imprimiendo las 5 primeras observaciones del dataframe
df_solar.head(5)

