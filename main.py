# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pandas_profiling import ProfileReport
# import datetime

# from quickda.explore_data import *
# from quickda.clean_data import *
# from quickda.explore_numeric import *
# from quickda.explore_categoric import *
# from quickda.explore_numeric_categoric import *
# from quickda.explore_time_series import *

# #PREGUNTA 1_____________________________________________________
# data = pd.read_csv("baseball_reference_2016_scrape.csv")

# # Obtener los nombres de las columnas
# head = data.head()
# print(head)
# print("\n")


# # Obtener información general sobre el conjunto de data
# print(data.info())
# print("\n")

# # Obtener un resumen estadístico de las columnas numéricas
# print(data.describe())
# print("\n")

# #Obtener el número de filas y columnas
# print(data.shape)
# print("\n")

# #Explore with quickda
# print(explore(data, method="summarize"))


# #PREGUNTA 6_____________________________________________________

# #Limpiar datos

# # formato hora
# def to_time(start_time):
#     start_time = start_time.replace("Start Time: ", "")
#     start_time = start_time.replace(":", " ")
#     hour, minute, period, local = [x.strip() for x in start_time.split(" ")]
#     hour = int(hour)
#     minute = int(minute)
#     if period == "p.m." and hour != 12:
#         hour += 12
#     elif period == "a.m." and hour == 12:
#         hour = 0
#     return datetime.time(hour, minute)


# #Attendance a integer
# data['attendance'] = data['attendance'].str.strip("']").str.replace(',','')
# data = data[pd.to_numeric(data['attendance'], errors='coerce').notnull()]
# data['attendance'] = pd.to_numeric(data['attendance'])

# #Quitar columnas que no aportan información
# data.dropna(axis=1, how='all', inplace=True) #todas columasn vacias
# data.drop(['other_info_string'], axis=1, inplace=True) #vacia
# data.drop(['boxscore_url'], axis=1, inplace=True) #url no aporta info

# #Pasar duracion a minutos
# data['game_duration'] = data['game_duration'].str.replace(':','').astype(int)
# data['game_duration'] = data['game_duration'].apply(lambda x: x/100*60 + x%100)

# #Quitar los : de la columna venue
# data['venue'] = data['venue'].str.replace(':','')

# #cambiar tipo de dato de la fecha
# data['date'] = pd.to_datetime(data['date'])

# #cambiar formato de la hora
# data['start_time'] = data['start_time'].apply(to_time)

# data.to_csv("modificada.csv", index=False)

# #exploarar nuevos datos
# data = pd.read_csv("modificada.csv")
# print(explore(data, method="summarize"))


# #Explorar datos

# # Home hits vs away hits
# sns.scatterplot(x="home_team_hits", y="away_team_hits", data=data)
# plt.xlabel("Home Team Hits")
# plt.ylabel("Away Team Hits")
# plt.show()

# # Home runs vs away runs
# sns.scatterplot(x="home_team_runs", y="away_team_runs", data=data)
# plt.xlabel("Home Team Runs")
# plt.ylabel("Away Team Runs")
# plt.show()

# # Home errors vs away errors
# sns.scatterplot(x="home_team_errors", y="away_team_errors", data=data)
# plt.xlabel("Home Team Errors")
# plt.ylabel("Away Team Errors")
# plt.show()

# # Attendance vs game tye
# sns.barplot(x="game_type", y="attendance", data=data)
# plt.xlabel("Game Type")
# plt.ylabel("Attendance")
# plt.xticks(rotation=90)
# plt.show()


# #Attendance vs local team
# sns.barplot(x="home_team", y="attendance", data=data)
# plt.xlabel("Local Team")
# plt.ylabel("Attendance")
# plt.xticks(rotation=90)
# plt.show()

# #Attendance vs away team
# sns.barplot(x="away_team", y="attendance", data=data)
# plt.xlabel("Away Team")
# plt.ylabel("Attendance")
# plt.xticks(rotation=90)
# plt.show()

# #Most common home team
# team_counts = data["home_team"].value_counts()
# sns.barplot(x=team_counts.index, y=team_counts.values)
# plt.xlabel("Home Team")
# plt.ylabel("Numero de partidos")
# plt.xticks(rotation=90)
# plt.show()

# #Most common away team
# team_counts = data["away_team"].value_counts()
# sns.barplot(x=team_counts.index, y=team_counts.values)
# plt.xlabel("Home Team")
# plt.ylabel("Numbero de partidos")
# plt.xticks(rotation=90)
# plt.show()



# #PREGUNTA 4_____________________________________________________

# # separar las variables numéricas y categóricas
# num_vars = data.select_dtypes(include=['int', 'float'])
# cat_vars = data.select_dtypes(exclude=['int', 'float'])

# # matriz de correlación
# sns.heatmap(num_vars.corr(), annot=True)
# plt.show()

# #pandas profiling report
# profile = ProfileReport(data, title="Pandas Profiling Report")
# profile.to_file("report.html")



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importar el conjunto de datos

datos = pd.read_csv('modificada.csv')
X = datos.iloc[:, :-1]
y = datos.iloc[:, -1]

print(X)
print(y)

#codificar las variables categoricas

def codif_y_ligar(dataframe_original, variables_a_codificar):
    dummies = pd.get_dummies(dataframe_original[[variables_a_codificar]])
    res = pd.concat([dataframe_original, dummies], axis = 1)
    res = res.drop([variables_a_codificar], axis = 1)
    return(res) 

variables_a_codificar = ['away_team', 'game_type', 'home_team']   #  Esta es una lista de variables
for variable in variables_a_codificar:
    X = codif_y_ligar(X, variable)

### Codificar la variable dependiente
y = pd.get_dummies(y)
print(y)


#quitar columnas innecesarias
X = X.drop(['date', 'start_time'], axis = 1)
print(X)

## División del conjunto de datos en un conjunto para entrenamiento y otro para pruebas

from sklearn.model_selection import train_test_split
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 1/3, random_state = 0)

print(X_entreno)
print(X_prueba)
print(y_entreno)
print(y_prueba)

#escalamiento de caracteristicas
from sklearn.preprocessing import MinMaxScaler

escalador = MinMaxScaler()
X_entreno['attendance'] = escalador.fit_transform(X_entreno['attendance'].values.reshape(-1,1))
print(X_entreno)

## Entrenamiento del modelo de regresión lineal simple con el conjunto de datos para entrenamiento
from sklearn.linear_model import LinearRegression
regresor_lin = LinearRegression()
regresor_lin.fit(X_entreno, y_entreno)


from sklearn.preprocessing import PolynomialFeatures
# Parte 1
regresor_poli = PolynomialFeatures(degree = 2)
X_poli = regresor_poli.fit_transform(X)
# Parte 2
regresor_lin_2 = LinearRegression()
regresor_lin_2.fit(X_poli, y)

print("-----------------------------------------")

print(X_entreno.size)
print(y_entreno.size)

#Visualización de resultados con Regresión Lineal
plt.scatter(X_entreno, y_entreno, color = 'red')
plt.plot(X_entreno, regresor_lin.predict(X_entreno), color = 'blue')
plt.title('Verdad o Mentira (Regresion Lineal)')
plt.xlabel('Nivel de Posición')
plt.ylabel('Salario')
plt.show()


#Visualización de resultados con Regresión Polinomial

plt.scatter(X_entreno, y_entreno, color = 'red')
plt.plot(X_entreno, regresor_lin_2.predict(regresor_poli.fit_transform(X_entreno)), color = 'blue')
plt.title('Verdad o Mentira (Regresión Polinomial)')
plt.xlabel('Nivel del Puesto')
plt.ylabel('Salario')
plt.show()

#Predicción de los resultados con Regresión Lineal
regresor_lin.predict([[6.5]])

#Predicción de los resultados con Regresión Polinomial
regresor_lin_2.predict(regresor_poli.fit_transform([[6.5]]))