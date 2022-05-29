CREATE TABLE weather (
    id int auto_increment primary key,
    date DATETIME,
    main VARCHAR(30),
    temp float,
    feels_like float,
    wind_speed float
);

CREATE TABLE distance (
    station_id VARCHAR(50) primary key,
    station_lat double,
    station_lon double,
    station_name VARCHAR(50),
    mt1 int,
    cs2 int,
    sc4 int,
    ac5 int,
    sw8 int,
    ct1 int
);

CREATE TABLE bike (
    id int auto_increment primary key,
    rack_tot_cnt int,
    parking_bike_tot_cnt int,
    shared int,
    station_id VARCHAR(50),
    date DATETIME,
    foreign key (station_id) references distance(station_id) on update cascade on delete restrict
);

CREATE TABLE predict_all (
    station_id VARCHAR(50),
    station_lat double,
    station_lon double,
    now int,
    thirty int,
    one int,
    two int,
    time DATETIME
);

CREATE TABLE now_all (
    station_id VARCHAR(50),
    station_lat double,
    station_lon double,
    now int,
    thirty int,
    one int,
    two int,
    time DATETIME
);
