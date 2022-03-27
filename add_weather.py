from pip import main
import pymysql
# import pandas as pd
# import numpy as np
import requests
import json

conn = pymysql.connect(host='localhost',
                       user='yusun',
                       password='1234',
                       db='weather',
                       charset='utf8')

sql = "INSERT INTO user (name, email) VALUES (%s, %s)"

API_KEY = "3d3df4ac693138f8157192b265f0844e"
lon = 127.10
lat = 37.51
weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric'


resp = requests.get(weather_url)
weather_json = json.loads(resp.text)

print(weather_json)
print('\n')
print(weather_json['name'])               # 호칭
print(weather_json['weather'][0]['main'])    # 흐림/ 맑음 등
print(weather_json['main'])
print(weather_json['main']['temp'])       # 기온
print(weather_json['main']['feels_like']) # 체감온도
print(weather_json['wind']['speed'])      # 풍속

