from pyspark import SparkContext
from pathlib import Path
import os

def parse_record(record):
    date =  record[15:23]
    latitude = record[28:34]
    longitude = record[34:41]
    wind_speed = int(record[65:69])
    air_temp = int(record[87:92])
    return (date[0:4],(date,latitude,longitude,wind_speed,air_temp))


sc = SparkContext(appName="weather")
# rdd = sc.textFile("/home/DATA/NOAA_weather/[1980-2010]/*.gz")
# read the file from 1980-2000
rdd = sc.textFile("./{1999,2000}/*.gz")
all_fields = rdd.map(parse_record)

# get min, max of temperature and max wind speed of year
max_wind_year = all_fields.map(lambda x: (x[0],x[1][3]))\
    .filter(lambda x: x[1]!=9999)\
    .reduceByKey(lambda x,y: max(x,y))
min_temp_year = all_fields.map(lambda x: (x[0],x[1][4]))\
    .filter(lambda x: x[1]!=9999)\
    .reduceByKey(lambda x,y: min(x,y))
max_temp_year = all_fields.map(lambda x: (x[0],x[1][4]))\
    .filter(lambda x: x[1]!=9999)\
    .reduceByKey(lambda x,y: max(x,y))
# print(max_temp_year.union(min_temp_year).reduceByKey(lambda x,y: (x,y)).union(max_wind_year).reduceByKey(lambda x,y: (*x,y)).collect())
# union the result and persist it
output1_name = "./output1"
output1 = Path(output1_name)
if output1.is_dir():
    os.rmdir(output1_name)
union_fields = max_temp_year.union(min_temp_year)\
    .reduceByKey(lambda x,y: (x,y)).union(max_wind_year)\
    .reduceByKey(lambda x,y: (*x,y)).coalesce(1).saveAsTextFile(path=output1_name, compressionCodecClass= "org.apache.hadoop.io.compress.GzipCodec")
