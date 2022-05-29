#!/usr/bin/env python
# coding: utf-8

# ## sql 정보를 보는 방법
# 
# 1. sql 에 접속을 먼저 해서 conn 을 만들어야 합니다. (conn = connect) 
# 2. sql에서 조회할 명령어 (query)를 conn으로 전달합니다.
# 3. 결과를 받고, conn을 닫습니다.
# 

# In[1]:


import pymysql
import pandas as pd
import numpy as np
# import xgboost as xgb
# import matplotlib.pyplot as plt

from pandas import DataFrame
# from xgboost import plot_importance
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

import MySQLdb
import datetime


#  

# In[4]:


time = datetime.datetime.now() + datetime.timedelta(minutes=1)
time = datetime.datetime.strftime(time, "%Y-%m-%d %H:%M")


# In[5]:


time


# In[6]:


conn = pymysql.connect(
    host='172.18.0.2', 
    port=3306,
    user='root',
    passwd='insight2022',
    db='insight',
    charset='utf8',
    autocommit=False
)

cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)


# In[7]:



distance = "select * from distance;"
cursor.execute(distance)
distance = pd.DataFrame.from_dict(cursor.fetchall())
distance = distance[['station_id','station_lat','station_lon']]
# distance


# In[8]:


# 예측값 뽑아내기 1

bike_30 = "select * from predict_thirty where time BETWEEN DATE_SUB(NOW(), INTERVAL 35 minute) and DATE_SUB(NOW(), INTERVAL 29 minute) "
cursor.execute(bike_30)
bike_30_df = pd.DataFrame.from_dict(cursor.fetchall())
# bike_30_df = bike_30_df[['station_id', 'result']]
# bike_30_df.columns = ['station_id', 'thirty']
# bike_30_df


# In[9]:


bike_30_df = bike_30_df[['station_id', 'result']]
bike_30_df.columns = ['station_id', 'thirty']
# bike_30_df


# In[10]:


# 예측값 뽑아내기 2

bike_1 = "select * from predict_one where time BETWEEN DATE_SUB(NOW(), INTERVAL 65 minute) and DATE_SUB(NOW(), INTERVAL 58 minute) "
cursor.execute(bike_1)
bike_1_df = pd.DataFrame.from_dict(cursor.fetchall())

# bike_1_df


# In[11]:


bike_1_df = bike_1_df[['station_id', 'result']]
bike_1_df.columns = ['station_id', 'one']
# bike_1_df


# In[12]:


# 예측값 뽑아내기 3

bike_2 = "select * from predict_two where time BETWEEN DATE_SUB(NOW(), INTERVAL 128 minute) and DATE_SUB(NOW(), INTERVAL 118 minute) "
cursor.execute(bike_2)
bike_2_df = pd.DataFrame.from_dict(cursor.fetchall())

# bike_2_df


# In[13]:


bike_2_df = bike_2_df[['station_id', 'result']]
bike_2_df.columns = ['station_id', 'two']
# bike_2_df


# In[14]:


bike = "select * from bike where date between date_sub(now(), interval 6 minute) and now();"
cursor.execute(bike)
bike_df = pd.DataFrame.from_dict(cursor.fetchall())

# bike_df


# In[15]:


bike_df = bike_df[['station_id','parking_bike_tot_cnt']]
bike_df.columns = ['station_id','now']
# bike_df


# In[16]:


# 3. conn 연결 해제 (마지막 1회 수행)
cursor.close()
conn.close()


# In[17]:


merge_table = pd.merge(distance, bike_df, how='inner', on='station_id') 
merge_table = pd.merge(merge_table, bike_30_df, how='inner', on='station_id') 
merge_table = pd.merge(merge_table, bike_1_df, how='inner', on='station_id') 
merge_table = pd.merge(merge_table, bike_2_df, how='inner', on='station_id') 
merge_table['time'] = time
merge_table


# In[18]:


engine = create_engine("mysql+mysqldb://root:insight2022@172.18.0.2:3306/insight", encoding='utf-8')
conn = engine.connect()
merge_table.to_sql(name='now_all', con=engine, if_exists='append', index=False)

conn.close()


# In[ ]:




