from asyncio.log import logger
import pymysql
import requests
import json
import pandas as pd
from typing import Any, Dict, List
from datetime import date, datetime, timezone, timedelta

# Insert 쿼리
QUERY = '''INSERT INTO weather
(
    date,
    main,
    temp,
    feels_like,
    wind_speed
)
values
(
    %(date)s,
    %(main)s,
    %(temp)s,
    %(feels_like)s,
    %(wind_speed)s
);
'''


# 날씨 호출

def call_weather() -> Dict[str, Any]:
    """open weather api에서 데이터를 받아옵니다.
       dict 형태로 변환합니다.

    Arguments:

    Returns:
        dict(str, Any): 변환된 CSV 데이터
    """
    LON = 126.9896
    LAT = 37.5335
    WEATHER_API_KEY = '3d3df4ac693138f8157192b265f0844e'

    weather_url = f'https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={WEATHER_API_KEY}&units=metric'
    resp = requests.get(weather_url)
    weather_json = json.loads(resp.text)

    dict_ = {}
    KST = timezone(timedelta(hours=9))

    dict_['date'] = str(datetime.now(KST)).split(".")[0]      # 날짜
    dict_['main'] = weather_json['weather'][0]['main']              # 흐림/ 맑음 등
    dict_['temp'] = weather_json['main']['temp']                    # 기온
    dict_['feels_like'] = weather_json['main']['feels_like']        # 체감온도
    dict_['wind_speed'] = weather_json['wind']['speed']             # 풍속

    return dict_


def main() -> None:
    data = call_weather()

    
    try:
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
        cursor.execute(QUERY, data)
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

