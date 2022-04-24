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

# Select 쿼리
QUERY_ID = 'select station_id from distance;'


# distance 테이블의 station_id 확인
def select_valid(data: pd.DataFrame, station_id: List) -> List[Dict[str, Any]]:
    """distance 테이블의 station_id와 bike_data의 station_id 가 일치하는 것만 추출합니다.

    Arguments:
        data : 비교할 데이터 프레임 (api호출로 인한)
        station_id : 비교할 distance의 station_id


    Returns:
        dict(str, Any) : 두 데이터에 모두 있는 데이터만을 가진 dictionary
    """

    for val_bike in data['station_id'].to_list():
        exist = 0
        for val_id in station_id:
            if val_bike == val_id['station_id']:
                exist = 1
                break
        if exist == 0:
            index = data[data['station_id'] == val_bike].index
            data.drop(index, inplace=True)    

    data = data.to_dict('records')
    return data


# 따릉이 데이터 호출
def call_bike() -> pd.DataFrame:
    """open weather api에서 데이터를 받아옵니다.
       dict 형태로 변환합니다.

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
    
    bike_data.reset_index(drop=True, inplace=True)

    return bike_data





def main() -> None:
    try:
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
        cursor.execute(QUERY_ID)
        station_id = cursor.fetchall()

        data = select_valid(data=data, station_id=station_id)


        # for value in data:
        cursor.executemany(QUERY, data)
        conn.commit()
        cursor.close()
        conn.close()

    except pymysql.err.OperationalError:
        print("[OperationalError] : 데이터소스 이름 없음, 트랜젝션 실행불가, 메모리 할당 에러, 연결 끊김 등 에러")
    except pymysql.err.IntegrityError:
        print("[IntegrityError] : 커서 유효하지 않음, 트랜젝션 안맞음 등 에러")
    except pymysql.err.InterfaceError:
        print("[InterfaceError] : 데이터베이스 자체 말고 인터페이스 문제, eg.) interface 할 때 컬럼 개수와 value 개수가 안맞음.")



if __name__ == '__main__':
    main()
