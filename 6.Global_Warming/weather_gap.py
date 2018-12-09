import pandas as pd
import numpy as np


df = pd.read_csv("output.csv", header=0)
df_date = df.where(df['date']<=1989).groupby(['lat','lon']).agg({"avg(wind_speed)":"mean","avg(temp)":"mean"})
df_date = df_date.dropna()

df_date2 = df.where(df['date']>=2000).groupby(['lat','lon']).agg({"avg(wind_speed)":"mean","avg(temp)":"mean"})
df_date2 = df_date2.dropna()

df_out = pd.merge(df_date,df_date2,on=['lat','lon'],how='inner')
df_out['gap_temp'] = df_out['avg(temp)_y'] - df_out['avg(temp)_x']
df_out['gap_wind'] = df_out['avg(wind_speed)_y'] - df_out['avg(wind_speed)_x']
df_out.to_csv('gap.csv')

