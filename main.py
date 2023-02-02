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


# # #Attendance a integer
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

# #pasar a fecha 0/0/0 a dia de semana int
# data['date'] = data['date'].apply(lambda x: x.weekday())

# #cambiar formato de la hora
# data['start_time'] = data['start_time'].apply(to_time)

# #pasar hora de 00:00:00 a solo hora
# data['start_time'] = data['start_time'].apply(lambda x: x.hour)

# #make attendence the last column
# cols = list(data.columns.values)
# cols.pop(cols.index('attendance'))
# data = data[cols+['attendance']]

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

datos = pd.read_csv('modificada.csv')
#drop unnecesary columns
datos.drop(['away_team_errors'], axis=1, inplace=True) #no aporta info
datos.drop(['home_team_errors'], axis=1, inplace=True) #no aporta info
datos.drop(['away_team_hits'], axis=1, inplace=True) #no aporta info
datos.drop(['home_team_hits'], axis=1, inplace=True) #no aporta info
datos.drop(['away_team_runs'], axis=1, inplace=True) #no aporta info
datos.drop(['home_team_runs'], axis=1, inplace=True) #no aporta info
datos.drop(['game_duration'], axis=1, inplace=True) #no aporta info

X = datos.iloc[:, :-1]
y = datos.iloc[:, -1]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dummies = ['home_team', 'away_team', 'venue', 'game_type']
position = [datos.columns.get_loc(c) for c in dummies if c in datos]

# columntransformer all dummie positions
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), position)], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#______________________________________________________________

from sklearn.model_selection import train_test_split
X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(X_entreno, y_entreno)

y_pred = regresor.predict(X_prueba)
np.set_printoptions(precision=2)

print(X)

print(regresor.predict([[1, 0, 0, 160000, 130000, 300000]]))

print(regresor.coef_)
print(regresor.intercept_)