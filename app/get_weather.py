import pandas as pd
import numpy as np

import urllib2

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

# # download weather data
# for y in range(2000,2016):
#     for m in range(1,13):
#         print y,m
        
#         url = 'http://www.wunderground.com/history/airport/KSFO/{year}/{month}/1/MonthlyHistory.html?req_city=San%20Francisco&req_state=CA&req_statename=California&reqdb.zip=94101&reqdb.magic=1&reqdb.wmo=99999&format=1'.format(year=y, month=m)

#         response = urllib2.urlopen(url)
#         html = response.read()

#         f = open('SF-weather_{year}-{month}.csv'.format(year=y, month=m), 'w')
#         f.write(html)
#         f.close()

# clean up the csv files
for y in range(2000,2016):
    for m in range(1,13):
        print y,m

        filename = 'SF-weather/SF-weather_{year}-{month}.csv'.format(year=y, month=m)
        with open(filename, 'r') as fin:
            filedata = fin.read()
            with open(filename.replace('.csv', '_.csv'), 'w') as fout: 
                fout.write(filedata.replace(', ', ',').replace('<br />', '').replace('PST','PDT'))

# put the csv files into dataframe        
dfs = []
for y in range(2000,2016):
    for m in range(1,13):
        print y,m

        df = pd.read_csv('SF-weather/SF-weather_{year}-{month}_.csv'.format(year=y, month=m))
        df.PDT = pd.to_datetime(df.PDT)
        dfs.append(df)

dff = pd.concat(dfs)


###############################################################################
# Convert pandas to sql

dbname = 'photo_db'
username = 'ysakamoto'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine.url

if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))


dff.to_sql('sf_weather', engine, if_exists='replace')

###############################################################################
# connect:
con = None
con = psycopg2.connect(database = dbname, user = username)

sql_query="""
SELECT "PDT", "Events" from sf_weather;
""" 
weather = pd.read_sql_query(sql_query,con)
print weather.shape

# get photo data

sql_query="""
SELECT DISTINCT * from photo_data_table 
WHERE DATE_PART('year', datetaken) >1999
;
""" 
photos = pd.read_sql_query(sql_query,con)
print photos.shape

con.close()

# get weather of the datetaken
weather.PDT = weather.PDT.apply(lambda x: x.date())
weather_dict = weather.set_index('PDT').to_dict()
weather['Events'] = weather['Events'].fillna('Sunny')

photo_weather = photos.datetaken.apply(lambda x: weather_dict['Events'][x.date()])

photos['weather'] = photo_weather
photos['weather']= photos['weather'].fillna('Sunny')

import matplotlib.pyplot as plt
import seaborn as sb


days_weather = weather.groupby('Events').count()
photos_weather = photos.groupby('weather')['id'].count()

photo_ave = photos_weather/days_weather.transpose()

photo_ave.transpose().plot(kind='bar', colors=sb.color_palette(), figsize=(5,5), fontsize=12)
plt.xticks(np.arange(0,6,1), ['Fog', 'Fog/Rain', 'Rain', 'Rain/Tstorm', 'Sunny', 'Tstorm'])
plt.xlabel('')
# plt.xticks(range(1,3), ['', ''])
plt.legend('')
# plt.xlim(-0.5,1)
plt.tight_layout()
plt.savefig('weather.png')
plt.show()

