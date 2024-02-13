##### Análisis de Indicadores del Banco Mundial  
##### INDICADOR DE ACCESO A ELECTRICIDAD (% DE LA POBLACION)
##### Autora:Pamela Paredes

### Librerias:
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn 

### Base de datos
## Base de datos de paises y regiones (1):
df_countries = pd.read_excel("Data/acceso_electr_BM.xls", sheet_name="Metadata - Countries")
print (df_countries)

## Base de datos del índice de acceso a electricidad (2):
df_acceso_electr= pd.read_excel("Data/acceso_electr_BM.xls",sheet_name="Data", skiprows=3)
print(df_acceso_electr)

### PREGUNTA 1: ¿Cuál es el valor promedio del acceso a electricidad entre los países de América Latina en el año 2020?

## Filtración de la base (1) y unión con la base (2):
america_latina= df_countries[df_countries["Region"]=="América Latina y el Caribe (excluido altos ingresos)"]
df_AL_electr = america_latina.merge(df_acceso_electr,on=["Country Name","Country Code"], how="left")
print (df_AL_electr)

## Cálculo del acceso a electricidad promedio para año 2020
promedio_AL_2020=df_AL_electr["2020"].mean() 
print(promedio_AL_2020)

## Análisis:
print(f"El promedio de acceso a electricidad de los países de América Latina en 2020 es de {promedio_AL_2020} %.")
print(f"Es decir,en promedio el {promedio_AL_2020} % de la población de los países de América Latina tienen acceso a electricidad.")


### PREGUNTA 2: ¿Cómo ha evolucionado el indicador de acceso a electricidad a lo largo del tiempo en América Latina?

## Elimina del df las variables cualitativas
df_AL_electr_1 = df_AL_electr.drop(["Country Name", "Country Code", "Region", "Income_Group", "Indicator Name", "Indicator Code"], axis=1)

## Resumen de las variables cuantitativas del dataframe
resumen_AL= df_AL_electr_1.describe()
print (resumen_AL)

## Extracción del vector de medias del indice de acceso a electricidad para América Latina
media_AL = resumen_AL.iloc[1].to_numpy(copy=True) #Python inicia el conteo desde cero.
media_AL

## Creación de un df con los años (desde 1960 hasta 2022) y el vector de medias por año
#Creando el df
df_serie_AL=pd.DataFrame()
df_serie_AL

#Creando el vector de años
anios = list(range(1960,2023,1)) 

#Llenando el df
df_serie_AL["Year"]=anios
df_serie_AL["Media AL"]=media_AL
print(df_serie_AL)

## Eliminando los valores nulos para graficar solo a partir del año que existen valores (1990):
df_serie_AL=df_serie_AL.dropna()
print(df_serie_AL)

## Gráfico del promedio de acceso a electricidad para América Latina en el tiempo:
Graf_AL=sn.lineplot(data = df_serie_AL, x="Year", y="Media AL")
plt.title("Evolución del índice de acceso a electricidad de Amércia Latina 1990-2021")
plt.show()

## Análisis:
# Existe una tendencia creciente en el acceso a electricidad promedio de América Latina a lo largo del tiempo (1990-2021).
# Adicional, desde 1990 hasta 1995 se observan picos irregulares en el promedio de acceso a electricidad en Amércia Latina. Esto quiere decir que entre estos años, existieron disminuciones del porcentaje de la población que tenía acceso a electricidad, el cual puede deberse a varios factores, entre estos el aumento de la población.



### PREGUNTA 3: ¿Cómo es el mapa de correlación entre los últimos 5 años de datos disponibles para los países de América Latina?

## Extracción de los últimos 5 años del df para cada país de América Latina:
df_AL_5_years=df_AL_electr_1[["2017","2018","2019","2020","2021"]]
print(df_AL_5_years)

## Matriz de correlación:
matriz_corr=df_AL_5_years.corr()
print(matriz_corr)

## Mapa de calor de correlaciones:
mapa_corr= sn.heatmap(matriz_corr, annot=True, cmap="YlGnBu", vmax=1, vmin=-1)
plt.title("Mapa de correlacion")
plt.show()

## Análisis:
# Durante los últimos 5 años existe una fuerte correlación del uso de internet para los países de América Latina.

