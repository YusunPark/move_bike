#!/usr/bin/env python
# coding: utf-8

# # 데이터 준비
# 

# In[1]:


import pymysql
import pandas as pd
import numpy as np
import xgboost as xgb
# import matplotlib.pyplot as plt

# from pandas import DataFrame
# from xgboost import plot_importance

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

import MySQLdb
from datetime import datetime


# In[2]:


time = datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M")


# In[3]:


d_QUERY = "select * from distance;"


# # 30분 단위

# In[4]:


# 4일간 매 30분 - 10분마다 실행 (0, 30) (10, 40) (20, 50) 

thirty_QUERY = """
        SELECT * 
        FROM bike 
        WHERE (date BETWEEN DATE_SUB(NOW(), INTERVAL 4 DAY ) AND NOW())
        AND minute(date) in (round(minute(NOW()), -1), round(minute(DATE_ADD(NOW(), INTERVAL 30 MINUTE)), -1));
        """


# In[5]:


# 서버시간 맞추기 위해서
server = "select @@global.time_zone, @@session.time_zone,@@system_time_zone;"
server1 = "SET GLOBAL time_zone='+09:00';"
server2 = "SET time_zone='+09:00';"
server3 = "select now()"


# In[6]:


# DB에서 데이터 로딩 
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

cursor.execute(thirty_QUERY)
bike_data = cursor.fetchall() # list of dictionary
bike_df = pd.DataFrame.from_dict(bike_data)

cursor.execute(d_QUERY)
distance_data = cursor.fetchall() # list of dictionary
distance_df = pd.DataFrame.from_dict(distance_data)

cursor.close()
conn.close()


# In[7]:


# 모델 학습에 넣기 전에 형식 전처리
bike_df['date']=bike_df['date'].dt.strftime('%Y_%m_%d_%Hh%Mm')
bike_df = bike_df.drop_duplicates(['station_id', 'date'])


# In[8]:


# pivot을 통해서 데이터의 형태를 변경 (x,y 축 변경)
data = bike_df.pivot(index='station_id', columns='date', values='parking_bike_tot_cnt')


# In[9]:


# distance_df 에서 필요한 열만 추출
d = distance_df[distance_df.columns[[0,4,5,6,7,8,9]]]

#  d, data 를 합침
data = pd.merge(d , data, on='station_id', how='inner').set_index('station_id')


# ## X, Y 데이터 분리하기
# 

# In[10]:


data.isnull().sum()
data = data.fillna(0)


# In[11]:


# X, y 데이터 분리
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]


# In[12]:


# 라벨 값의 비율 확인 
# y.value_counts()


# ## train / test 분리
# 

# In[13]:


# train, test 7:3 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# In[14]:


# print(X_test.shape)
# print("Train ratio of X:", X_train.shape[0] / X.shape[0])
# print("Test ratio of X:", X_test.shape[0] / X.shape[0])


# In[15]:


# 거리데이터와 자전거 잔여대수 데이터의 스케일 조정
# X_train 기준으로 fit하고 나머지는 동일한 것으로 진행한다.

needScale = X_train.columns.to_list() # scale 이 필요한 컬럼들

MMS = MinMaxScaler()

for column in needScale:
    X_train[column] = MMS.fit_transform(X_train[column].to_numpy().reshape(-1, 1))
    X_test[column] = MMS.transform(X_test[column].to_numpy().reshape(-1, 1))


# ### DACON - XGboost
# 
# 

# In[16]:


reg = xgb.XGBRegressor(max_depth = 4, n_estimators = 600, learning_rate=0.1)
reg.fit(X_train,y_train, early_stopping_rounds= 100, eval_set=[(X_test, y_test)])


score = reg.score(X_train, y_train)   
print("Training score: ", score) 

pred_score = reg.score(X_test, y_test)   
print("Testing score: ", pred_score) 
 

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
# print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse**(1/2.0)))

# 숫자가 적을 수록 예측값과 실제 값의 차이가 적은 것을 의미함.
mae = mean_absolute_error(y_test, y_pred)
# print("mae: %.2f" % mae)


# In[17]:


# print("Testing score: ", pred_score) 
# print("RMSE: %.2f" % (mse**(1/2.0)))

# plot_importance(reg, height=0.9, max_num_features=15)
# plt.show()


# In[18]:


# DB에 넣을 pred값 생성을 위해서 제일 오래된 값 하나를 삭제
pred_X = data.drop(data.columns[6], axis=1)

needScale = pred_X.columns.to_list()# scale 이 필요한 컬럼들

for column in needScale:
    pred_X[column] = MMS.transform(pred_X[column].to_numpy().reshape(-1, 1))

y_pred = reg.predict(pred_X)
yy_pred = y_pred.round()

# 5분 뒤 값 예측
# yy_pred


# In[19]:


# 새로만들 컬럼의 설정을 정의


table = 'predict_thirty'
score_table = 'grade_thirty'



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


# 값 업데이트

adding_q = f"INSERT INTO {score_table}(train_score, test_score, rmse) VALUES({score},{pred_score},{mse**(1/2.0)})"
cursor.execute(adding_q)
  

# conn.commit()
cursor.close()
conn.close()


# In[20]:


pred_table =  y.reset_index()
pred_table = pred_table.drop(pred_table.columns[-1], axis=1)
pred_table['result'] = yy_pred
thirty_result = yy_pred
pred_table['time'] = time

# distance 테이블에 없는 station_id 는 제외
check_id = d[['station_id']]
ppred_table = pred_table.merge(check_id, how='left')

# 데이터 삽입
engine = create_engine("mysql+mysqldb://root:insight2022@172.18.0.2:3306/insight", encoding='utf-8')
conn = engine.connect()
ppred_table.to_sql(name=table, con=engine, if_exists='append', index=False)

conn.close()


# In[21]:


ppred_table.columns = ['station_id', 'thirty', 'time']
thirty_table = ppred_table


# # 1시간 단위

# In[22]:


# 일주일 1시간 단위 - 10분마다 실행 
one_QUERY = """
        SELECT * 
        FROM bike 
        WHERE (date BETWEEN DATE_SUB(NOW(), INTERVAL 7 DAY ) AND NOW())
        AND minute(date) = round(minute(NOW()), -1);
        """


# In[23]:


# DB에서 데이터 로딩 
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

cursor.execute(one_QUERY)
bike_data = cursor.fetchall() # list of dictionary
bike_df = pd.DataFrame.from_dict(bike_data)

cursor.execute(d_QUERY)
distance_data = cursor.fetchall() # list of dictionary
distance_df = pd.DataFrame.from_dict(distance_data)

cursor.close()
conn.close()


# In[ ]:





# In[24]:


# 모델 학습에 넣기 전에 형식 전처리
bike_df['date']=bike_df['date'].dt.strftime('%Y_%m_%d_%Hh%Mm')
bike_df = bike_df.drop_duplicates(['station_id', 'date'])


# In[25]:


# pivot을 통해서 데이터의 형태를 변경 (x,y 축 변경)
data = bike_df.pivot(index='station_id', columns='date', values='parking_bike_tot_cnt')


# In[26]:


# distance_df 에서 필요한 열만 추출
d = distance_df[distance_df.columns[[0,4,5,6,7,8,9]]]

#  d, data 를 합침
data = pd.merge(d , data, on='station_id', how='inner').set_index('station_id')


# ## X, Y 데이터 분리하기
# 

# In[27]:


data.isnull().sum()
data = data.fillna(0)


# In[28]:


# X, y 데이터 분리
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]


# In[29]:


# 라벨 값의 비율 확인 
# y.value_counts()


# ## train / test 분리
# 

# In[30]:


# train, test 7:3 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# In[31]:


# print(X_test.shape)
# print("Train ratio of X:", X_train.shape[0] / X.shape[0])
# print("Test ratio of X:", X_test.shape[0] / X.shape[0])


# In[32]:


# 거리데이터와 자전거 잔여대수 데이터의 스케일 조정
# X_train 기준으로 fit하고 나머지는 동일한 것으로 진행한다.

needScale = X_train.columns.to_list() # scale 이 필요한 컬럼들

MMS = MinMaxScaler()

for column in needScale:
    X_train[column] = MMS.fit_transform(X_train[column].to_numpy().reshape(-1, 1))
    X_test[column] = MMS.transform(X_test[column].to_numpy().reshape(-1, 1))


# ### DACON - XGboost
# 
# 

# In[33]:


reg = xgb.XGBRegressor(max_depth = 4, n_estimators = 600, learning_rate=0.1)
reg.fit(X_train,y_train, early_stopping_rounds= 100, eval_set=[(X_test, y_test)])


score = reg.score(X_train, y_train)   
# print("Training score: ", score) 

pred_score = reg.score(X_test, y_test)   
# print("Testing score: ", pred_score) 
 

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % (mse**(1/2.0)))

# 숫자가 적을 수록 예측값과 실제 값의 차이가 적은 것을 의미함.
mae = mean_absolute_error(y_test, y_pred)
# print("mae: %.2f" % mae)


# In[34]:


# print("Testing score: ", pred_score) 
# print("RMSE: %.2f" % (mse**(1/2.0)))

# plot_importance(reg, height=0.9, max_num_features=15)
# plt.show()


# In[35]:


# DB에 넣을 pred값 생성을 위해서 제일 오래된 값 하나를 삭제
pred_X = data.drop(data.columns[6], axis=1)

needScale = pred_X.columns.to_list()# scale 이 필요한 컬럼들

for column in needScale:
    pred_X[column] = MMS.transform(pred_X[column].to_numpy().reshape(-1, 1))

y_pred = reg.predict(pred_X)
yy_pred = y_pred.round()

# 5분 뒤 값 예측
# yy_pred


# In[ ]:





# In[36]:


# 새로만들 컬럼의 설정을 정의


table = 'predict_one'
score_table = 'grade_one'

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

# 값 업데이트
adding_q = f"INSERT INTO {score_table}(train_score, test_score, rmse) VALUES({score},{pred_score},{mse**(1/2.0)})"
cursor.execute(adding_q)
  

conn.commit()
cursor.close()
conn.close()


# In[37]:


pred_table =  y.reset_index()
pred_table = pred_table.drop(pred_table.columns[-1], axis=1)
pred_table['result'] = yy_pred
one_result = yy_pred
pred_table['time'] = time

# distance 테이블에 없는 station_id 는 제외
check_id = d[['station_id']]
ppred_table = pred_table.merge(check_id, how='left')

# # 데이터 삽입
engine = create_engine("mysql+mysqldb://root:insight2022@172.18.0.2:3306/insight", encoding='utf-8')
conn = engine.connect()
ppred_table.to_sql(name=table, con=engine, if_exists='append', index=False)

conn.close()


# In[38]:


ppred_table.columns = ['station_id', 'one', 'time']
one_table = ppred_table


# # 2시간 단위
# 

# In[39]:


from datetime import datetime

now = datetime.now()

if now.hour % 2 == 0:

    two_QUERY = """
        SELECT * 
        FROM bike 
        WHERE (date BETWEEN DATE_SUB(NOW(), INTERVAL 7 DAY ) AND NOW())
        AND hour(date) in (0,2,4,6,8,10,12,14,16,18,20,22)
        AND minute(date) = round(minute(NOW()), -1);
        """
else:
    # 일주일 2시간 단위 - 10분마다 실행
    two_QUERY = """
        SELECT * 
        FROM bike 
        WHERE (date BETWEEN DATE_SUB(NOW(), INTERVAL 7 DAY ) AND NOW())
        AND hour(date) in (1,3,5,7,9,11,13,15,17,19,21,23)
        AND minute(date) = round(minute(NOW()), -1);
        """


# In[40]:


# DB에서 데이터 로딩 
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

cursor.execute(two_QUERY)
bike_data = cursor.fetchall() # list of dictionary
bike_df = pd.DataFrame.from_dict(bike_data)

cursor.execute(d_QUERY)
distance_data = cursor.fetchall() # list of dictionary
distance_df = pd.DataFrame.from_dict(distance_data)

cursor.close()
conn.close()


# In[41]:


# 모델 학습에 넣기 전에 형식 전처리
bike_df['date']=bike_df['date'].dt.strftime('%Y_%m_%d_%Hh%Mm')
bike_df = bike_df.drop_duplicates(['station_id', 'date'])


# In[42]:


# pivot을 통해서 데이터의 형태를 변경 (x,y 축 변경)
data = bike_df.pivot(index='station_id', columns='date', values='parking_bike_tot_cnt')


# In[43]:


# distance_df 에서 필요한 열만 추출
d = distance_df[distance_df.columns[[0,4,5,6,7,8,9]]]

#  d, data 를 합침
data = pd.merge(d , data, on='station_id', how='inner').set_index('station_id')


# ## X, Y 데이터 분리하기
# 

# In[44]:


data.isnull().sum()
data = data.fillna(0)


# In[45]:


# X, y 데이터 분리
X = data.drop(data.columns[-1], axis=1)
y = data[data.columns[-1]]


# In[46]:


# 라벨 값의 비율 확인 
# y.value_counts()


# ## train / test 분리
# 

# In[47]:


# train, test 7:3 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# In[48]:


# print(X_test.shape)
# print("Train ratio of X:", X_train.shape[0] / X.shape[0])
# print("Test ratio of X:", X_test.shape[0] / X.shape[0])


# In[49]:


X_train.head()


# In[50]:


# 거리데이터와 자전거 잔여대수 데이터의 스케일 조정
# X_train 기준으로 fit하고 나머지는 동일한 것으로 진행한다.

needScale = X_train.columns.to_list() # scale 이 필요한 컬럼들

MMS = MinMaxScaler()

for column in needScale:
    X_train[column] = MMS.fit_transform(X_train[column].to_numpy().reshape(-1, 1))
    X_test[column] = MMS.transform(X_test[column].to_numpy().reshape(-1, 1))


# In[51]:


X_train.head()


# ### DACON - XGboost
# 
# 

# In[52]:


reg = xgb.XGBRegressor(max_depth = 4, n_estimators = 600, learning_rate=0.1)
reg.fit(X_train,y_train, early_stopping_rounds= 100, eval_set=[(X_test, y_test)])


score = reg.score(X_train, y_train)   
# print("Training score: ", score) 

pred_score = reg.score(X_test, y_test)   
# print("Testing score: ", pred_score) 
 

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
# print("MSE: %.2f" % mse)
# print("RMSE: %.2f" % (mse**(1/2.0)))

# 숫자가 적을 수록 예측값과 실제 값의 차이가 적은 것을 의미함.
# mae = mean_absolute_error(y_test, y_pred)
# print("mae: %.2f" % mae)


# In[53]:


# print("Testing score: ", pred_score) 
# print("RMSE: %.2f" % (mse**(1/2.0)))

# plot_importance(reg, height=0.9, max_num_features=15)
# plt.show()


# In[54]:


# DB에 넣을 pred값 생성을 위해서 제일 오래된 값 하나를 삭제
pred_X = data.drop(data.columns[6], axis=1)

needScale = pred_X.columns.to_list()# scale 이 필요한 컬럼들

for column in needScale:
    pred_X[column] = MMS.transform(pred_X[column].to_numpy().reshape(-1, 1))

y_pred = reg.predict(pred_X)
yy_pred = y_pred.round()

# 5분 뒤 값 예측
# yy_pred


# In[55]:


table = 'predict_two'
score_table = 'grade_two'


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

# 값 업데이트
 
adding_q = f"INSERT INTO {score_table}(train_score, test_score, rmse) VALUES({score},{pred_score},{mse**(1/2.0)})"
cursor.execute(adding_q)
  
conn.commit()
cursor.close()
conn.close()


# In[56]:


pred_table =  y.reset_index()
pred_table = pred_table.drop(pred_table.columns[-1], axis=1)
pred_table['result'] = yy_pred
two_result = yy_pred
pred_table['time'] = time

# distance 테이블에 없는 station_id 는 제외
check_id = d[['station_id']]
ppred_table = pred_table.merge(check_id, how='left')

# # 데이터 삽입
engine = create_engine("mysql+mysqldb://root:insight2022@172.18.0.2:3306/insight", encoding='utf-8')
conn = engine.connect()
ppred_table.to_sql(name=table, con=engine, if_exists='append', index=False)

conn.close()


# In[57]:


ppred_table.columns = ['station_id', 'two', 'time']
two_table = ppred_table


# In[58]:


pred_table =  y.reset_index()
pred_table.columns  = ['station_id', 'now']


# In[59]:


distance_df = distance_df[['station_id', 'station_lat', 'station_lon']]


# In[60]:


merge_table = pd.merge(pred_table, thirty_table, how='inner', on='station_id') 
merge_table = pd.merge(merge_table, one_table, how='inner', on=['station_id', 'time']) 
merge_table = pd.merge(merge_table, two_table, how='inner', on=['station_id', 'time']) 
merge_table = pd.merge(merge_table, distance_df, how='inner', on='station_id')


# In[ ]:





# In[61]:


engine = create_engine("mysql+mysqldb://root:insight2022@172.18.0.2:3306/insight", encoding='utf-8')
conn = engine.connect()
merge_table.to_sql(name='predict_all', con=engine, if_exists='append', index=False)

conn.close()


# In[62]:


merge_table


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




