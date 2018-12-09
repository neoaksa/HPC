### Purpose
Using pyspark to help anlysis the situaltion of global warming. The data is from NCDC(http://www.ncdc.noaa.gov/) through 1980 to 1989 and 2000 to 2009(except 1984 and 2004).

### Stretage
* Get the max/min temprature and max wind speed only filter mistaken data(9999). Steps are as follows:
  * Load files into RDD: `sc.textFile("/home/DATA/NOAA_weather/{198[0-3],198[5-9]}/*.gz")`
  * Extrat filds from files through map function: `parse_record`
  * Filter 9999 data: `.filter(lambda x: x[1]!=9999)`
  * reducebyKey to get max or min data ( the key is `year`): `.reduceByKey(lambda x,y: max(x,y)`
  
* Get the avgerage temprature and avg wind speed by year, latitude and longitude of station which is a fixed land station.
  * Load files into RDD. Same as mapreduce
  * Load RDD to Dataframe. `sqlContext.createDataFrame(all_fields,schema=["date","report_type","lat","lon","wind_speed","wind_qulity","temp"])`
  * Filter error data(9999) and station type(FM-12) `df.where((df['lat']!='+9999') & (df['lon']!='+9999') & (df['wind_speed']!=9999) & (df['temp']!=9999) & (df['report_type']=='FM-12'))`
  * aggragate average by year, latitude and longitude:`df.groupBy(['date',"lat","lon"]).agg({"wind_speed":"avg","temp":"avg"})`


### Result and visualization
* the max/min temprature and max wind speed
```
year, max_temp(10x), min_temp(10x), max_wind_speed(10x)
1980,600,-780,617
1981,580,-850,618
1983,616,-931,618
1984,617,-932,618
1982,617,-930,700
1986,607,-901,607
1987,607,-900,602
1985,611,-932,618
1989,606,-900,900
2000,568,-900,900
1988,607,-900,618
2002,568,-932,900
2001,568,-900,900
2003,565,-900,900
2005,610,-925,900
2006,610,-917,900
2007,610,-900,900
2008,610,-900,900
2009,610,-854,900
```
* the avgerage temprature and avg wind speed by year, latitude and longitude
