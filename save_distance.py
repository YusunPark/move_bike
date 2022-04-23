import csv
import pymysql
from typing import Any, Dict, List


# Insert 쿼리
QUERY = '''INSERT INTO distance
(
    station_id,
    station_lat,
    station_lon,
    station_name,
    mt1,
    cs2,
    sc4,
    ac5,
    sw8,
    ct1
)
values
(
    %(station_id)s,
    %(station_lat)s,
    %(station_lon)s,
    %(station_name)s,
    %(mt1)s,
    %(cs2)s,
    %(sc4)s,
    %(ac5)s,
    %(sw8)s,
    %(ct1)s
);
'''


def read_csv(file: str) -> List[Dict[str, Any]]:
    """CSV 파일을 읽어서 dict 형태로 변환합니다.

    Arguments:
        file(str): CSV 파일 이름

    Returns:
        dict(str, Any): 변환된 CSV 데이터
    """

    data: List[Dict[str, Any]] = []
    with open(file=file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            data.append(row)

    return data


def main() -> None:
    data = read_csv(file='final_distance_data.csv')

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

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
