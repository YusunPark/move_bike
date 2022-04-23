import pymysql
from datetime import date, datetime, timezone, timedelta
import requests
import json
import pandas as pd
from typing import Any, Dict, List



# Insert 쿼리
QUERY = '''INSERT INTO bike
(
    rack_tot_cnt,
    parking_bike_tot_cnt,
    shared,
    station_id,
    date
)
values
(
    %(rack_tot_cnt)s,
    %(parking_bike_tot_cnt)s,
    %(shared)s,
    %(station_id)s,
    %(date)s
);
'''


# 날씨 호출

def call_bike() -> List[Dict[str, Any]]:
    """open weather api에서 데이터를 받아옵니다.
       dict 형태로 변환합니다.

    Arguments:

    Returns:
        dict(str, Any): 변환된 CSV 데이터
    """
    BIKE_API_KEY = '74796e6d72736f6c39314a50736843'

    bike_url_1 = f'http://openapi.seoul.go.kr:8088/{BIKE_API_KEY}/json/bikeList/1/1000/'
    bike_url_2 = f'http://openapi.seoul.go.kr:8088/{BIKE_API_KEY}/json/bikeList/1001/2000/'
    bike_url_3 = f'http://openapi.seoul.go.kr:8088/{BIKE_API_KEY}/json/bikeList/2001/3000/'

    resp_bike_1 = requests.get(bike_url_1)
    bike_json_1 = json.loads(resp_bike_1.text)

    resp_bike_2 = requests.get(bike_url_2)
    bike_json_2 = json.loads(resp_bike_2.text)

    resp_bike_3 = requests.get(bike_url_3)
    bike_json_3 = json.loads(resp_bike_3.text)


    df1 = pd.DataFrame.from_dict(bike_json_1['rentBikeStatus']['row'])
    df2 = pd.DataFrame.from_dict(bike_json_2['rentBikeStatus']['row'])
    df3 = pd.DataFrame.from_dict(bike_json_3['rentBikeStatus']['row'])
    bike_data = pd.concat([df1, df2, df3])
    bike_data = bike_data[['rackTotCnt', 'parkingBikeTotCnt', 'shared', 'stationId']]
    bike_data.columns = ['rack_tot_cnt','parking_bike_tot_cnt','shared','station_id'] 
    
    KST = timezone(timedelta(hours=9))
    bike_data['date'] = str(datetime.now(KST)).split(".")[0]

    data = bike_data.to_dict('records')
    # print(data[:5])
    
    return data




def main() -> None:
    data = call_bike()

    conn = pymysql.connect(
        host='158.247.209.211', 
        port=3306,
        user='root',
        passwd='insight2022',
        db='insight',
        charset='utf8',
        autocommit=False
    )

    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

    # for value in data:
    cursor.executemany(QUERY, data)

    sql = 'select * from bike limit 3'
    cursor.execute(sql)
    a = cursor.fetchall()
    print(a)

    # conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
