from pyspark import SparkContext
from pathlib import Path
import shutil
from pyspark.sql import Row
from pyspark import SQLContext

def parse_record(record):
    date =  record[15:23]
    latitude = record[28:34]
    longitude = record[34:41]
    wind_speed = int(record[65:69])
    air_temp = int(record[87:92])
    return (date[0:4],(date,latitude,longitude,wind_speed,air_temp))

def parse_record_detail(record):
    date =  record[15:23]
    report_type = record[41:46]
    latitude = record[28:34]
    longitude = record[34:41]
    wind_speed = int(record[65:69])
    wind_qulity = record[69]
    air_temp = int(record[87:92])
    return (date[0:4],report_type,latitude,longitude,wind_speed,wind_qulity,air_temp)

def toCSV(record):
    output = record[0]
    output2 = ",".join(str(x) for x in record[1])
    return (output + "," + output2)

def f(x):
    d = {}
    for i in range(len(x)):
        d[str(i)] = x[i]
    return d


sc = SparkContext(appName="weather")
# read the file from 1980-2000
# rdd = sc.textFile("./{2000}/*.gz") #local
# rdd = sc.textFile("/home/DATA/NOAA_weather/{200[0-3],200[5-9]}/*.gz")
rdd = sc.textFile("/home/DATA/NOAA_weather/{198[0-3],198[5-9]}/*.gz")

# aggregation by year or date
is_agg_year = False

# union the result and persist it
output1_name = "./output1/"
output1_folder = Path(output1_name)
if output1_folder.is_dir():
    shutil.rmtree(output1_name)
# mapreduce way to get min, max of temperature and max wind speed of year
if is_agg_year:
    all_fields = rdd.map(parse_record)

    max_wind_year = all_fields.map(lambda x: (x[0],x[1][3]))\
        .filter(lambda x: x[1]!=9999)\
        .reduceByKey(lambda x,y: max(x,y))
    min_temp_year = all_fields.map(lambda x: (x[0],x[1][4]))\
        .filter(lambda x: x[1]!=9999)\
        .reduceByKey(lambda x,y: min(x,y))
    max_temp_year = all_fields.map(lambda x: (x[0],x[1][4]))\
        .filter(lambda x: x[1]!=9999)\
        .reduceByKey(lambda x,y: max(x,y))
    output1_rdd = max_temp_year.union(min_temp_year).reduceByKey(lambda x,y: (x,y)).union(max_wind_year).reduceByKey(lambda x,y: (*x,y)).map(toCSV)
    print(output1_rdd.collect())

    # print(max_wind_year.collect())
    output1_rdd.coalesce(1)\
        .saveAsTextFile(path=output1_name, compressionCodecClass= "org.apache.hadoop.io.compress.GzipCodec")
else:
    all_fields = rdd.map(parse_record_detail)
    sqlContext = SQLContext(sc)
    df = sqlContext.createDataFrame(all_fields,schema=["date","report_type","lat","lon","wind_speed","wind_qulity","temp"])
    df = df.where((df['lat']!='+9999') & (df['lon']!='+9999') & (df['wind_speed']!=9999) & (df['temp']!=9999)
                  & (df['report_type']=='FM-12'))
    df.groupBy(['date',"lat","lon"]).agg({"wind_speed":"avg","temp":"avg"}).coalesce(1).write.csv(output1_name,header=True)



# ~/spark/sbin/start-all.sh
# ~/spark/sbin/stop-all.sh
# ~/spark/bin/spark-submit --master=spark://arch06.cis.gvsu.edu:7077 infor_extra.py
#['1980,600,-780,617', '1981,580,-850,618'] ['1983,616,-931,618', '1984,617,-932,618', '1982,617,-930,700']
#['1986,607,-901,607', '1987,607,-900,602', '1985,611,-932,618']['1989,606,-900,900', '2000,568,-900,900', '1988,607,-900,618']
#['2002,568,-932,900', '2001,568,-900,900']['2003,565,-900,900', '2005,610,-925,900']
#['2006,610,-917,900', '2007,610,-900,900']['2008,610,-900,900', '2009,610,-854,900']






