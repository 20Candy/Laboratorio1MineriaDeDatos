import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import datetime

from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *

#PREGUNTA 1_____________________________________________________
data = pd.read_csv("baseball_reference_2016_scrape.csv")

# Obtener los nombres de las columnas
head = data.head()
print(head)
print("\n")


# Obtener información general sobre el conjunto de data
print(data.info())
print("\n")

# Obtener un resumen estadístico de las columnas numéricas
print(data.describe())
print("\n")

#Obtener el número de filas y columnas
print(data.shape)
print("\n")

#Explore with quickda
print(explore(data, method="summarize"))


#PREGUNTA 2_____________________________________________________

#Limpiar datos


# formato hora
def to_time(start_time):
    start_time = start_time.replace("Start Time: ", "")
    start_time = start_time.replace(":", " ")
    hour, minute, period, local = [x.strip() for x in start_time.split(" ")]
    hour = int(hour)
    minute = int(minute)
    if period == "p.m." and hour != 12:
        hour += 12
    elif period == "a.m." and hour == 12:
        hour = 0
    return datetime.time(hour, minute)


#Attendance a integer
data['attendance'] = data['attendance'].str.strip("']").str.replace(',','')
data = data[pd.to_numeric(data['attendance'], errors='coerce').notnull()]
data['attendance'] = pd.to_numeric(data['attendance'])

#Quitar columnas que no aportan información
data.dropna(axis=1, how='all', inplace=True) #todas columasn vacias
data.drop(['other_info_string'], axis=1, inplace=True) #vacia
data.drop(['boxscore_url'], axis=1, inplace=True) #url no aporta info

#Pasar duracion a minutos
data['game_duration'] = data['game_duration'].str.replace(':','').astype(int)
data['game_duration'] = data['game_duration'].apply(lambda x: x/100*60 + x%100)

#Quitar los : de la columna venue
data['venue'] = data['venue'].str.replace(':','')

#cambiar tipo de dato de la fecha
data['date'] = pd.to_datetime(data['date'])

#cambiar formato de la hora
data['start_time'] = data['start_time'].apply(to_time)

data.to_csv("modificada.csv", index=False)

#exploarar nuevos datos
data = pd.read_csv("modificada.csv")
print(explore(data, method="summarize"))








