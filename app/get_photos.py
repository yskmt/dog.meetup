"""

Flickr API:
https://www.flickr.com/services/api/

Flickr API limit: under 3600 queries per hour

Flickr API only allows maximum 40 pages. After that it gives back duplicates
https://stackoverflow.com/questions/1994037/flickr-api-returning-duplicate-photos/1996378#1996378

# launch postgres
postgres -D /usr/local/var/postgres

# start psql and connect to photo database:
psql -h localhost
\c photo_db



"""
import pdb

import flickrapi

from PIL import Image
import urllib2 as urllib
import io

import urllib, cStringIO
import PIL.Image as Image

import pandas as pd

import time
import datetime

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

import matplotlib.pyplot as plt
import seaborn as sb

import numpy as np

api_key = u'264d5eb00e8ca0a446d714cad1e4a99a'
api_secret = u'79f9b05a9003ace7'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

# photos = flickr.photos.search(user_id='44247215@N00', per_page='10')
# sets = flickr.photosets.getList(user_id='44247215@N00')

# Austin, TX
latlon = '30.285990, -97.739331'
# NYC, NY
# latlon = '40.756303, -73.985246'
# Palo Alto, CA
latlon = '37.426175, -122.141189'

latlon = map(lambda x: x.strip(), latlon.split(','))
lat = latlon[0]
lon = latlon[1]

bbox = '-122.511080,37.712002,-122.381984,37.809776'


def get_pics(year, bbox):

    d_min = datetime.date(year, 1, 1)
    d_max = datetime.date(year, 12, 31)
    
    utime_min = time.mktime(d_min.timetuple())
    utime_max = time.mktime(d_max.timetuple())

    dfs = []
    # maximum 40 pages
    # per page can go up to 500
    # use bbox
    
    for i in xrange(1, 41):
        print 'page: ', i

        # geo_context -  0: not defined, 1: indoors, 2: outdoors

        # lat=lat, lon=lon, radius=19
        photos = flickr.photos.search(bbox=bbox,
                                      content_type=1,
                                      text='dog',
                                      min_upload_date=str(utime_min),
                                      max_upload_date=str(utime_max),
                                      extras='geo,tags,url_t,url_m,url_o,'\
                                      'description,owner_name,views,path_alias,'\
                                      'date_upload,date_taken,machine_tags',
                                      per_page='250', page=i)

        # check if there is any results
        if len(photos['photos']['photo'])==0:
            break

        df_photos = pd.DataFrame(photos['photos']['photo'])

        # convert id to int and set it to index
        df_photos['id'] = df_photos['id'].astype(int)
        df_photos.set_index('id', inplace=True)

        # access to photo page
        # "https://www.flickr.com/photos/{user_id}/{photo_id}".format(user_id=df_photos.owner.iloc[0], photo_id=df_photos.index[0])
        
        # convert to int
        # height_o, pathalias, url_o, width_o can be null (_o for original)
        df_photos[['accuracy','context',
                   'datetakengranularity','datetakenunknown',
                   'farm',
                   'geo_is_contact','geo_is_family',
                   'geo_is_friend','geo_is_public',
                   'height_t','height_m','height_o',
                   'isfamily', 'ispublic', 'isfriend',
                   'server','views',
                   'width_t','width_m','width_o',
                   'woeid']]\
                   = df_photos[['accuracy','context',
                                'datetakengranularity','datetakenunknown',
                                'farm',
                                'geo_is_contact','geo_is_family',
                                'geo_is_friend','geo_is_public',
                                'height_t','height_m','height_o',
                                'isfamily', 'ispublic', 'isfriend',
                                'server','views',
                                'width_t','width_m','width_o',
                                'woeid']].fillna(0).astype(int)
            
        # just take the content of the dictionary
        df_photos['description'] \
            = df_photos['description'].apply(lambda x: x['_content'])

        # convert to datetime format
        df_photos['datetaken'] = pd.to_datetime(df_photos['datetaken'])
        df_photos['dateupload'] \
            = pd.to_datetime(df_photos['dateupload'].astype(int), unit='s')

        # convert latitude and longitude to float
        df_photos['latitude'] = df_photos['latitude'].astype(float)
        df_photos['longitude'] = df_photos['longitude'].astype(float)

        dfs.append(df_photos)

    if len(dfs)>0:
        dfs = pd.concat(dfs)
        dfs.to_csv('photos-%d.csv' %year, encoding='utf-8')

    return dfs



###############################################################################
# Convert pandas to sql

dbname = 'photo_db'
username = 'ysakamoto'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine.url

if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))

# 2001 - 2016
for yr in xrange(2000, 2010):
    print 'year', yr
    dfs = get_pics(yr, bbox)

    if len(dfs)>0:
        dfs.to_sql('photo_data_table', engine, if_exists='append')

    
# for photo_info in photos['photos']['photo']:

#     if len(photo_info['tags'].split(' '))<2:
#         continue
    
#     print photo_info['latitude'], photo_info['longitude']
#     print photo_info['url_m']
#     print photo_info['tags'].split(' ')

    # fd = urllib.urlopen(photo_info['url_t'])
    # image_file = io.BytesIO(fd.read())
    # im = Image.open(image_file)
    # im.show()
    

# file = cStringIO.StringIO(urllib.urlopen(photo_url).read())
# img = Image.open(file)



# connect:
con = None
con = psycopg2.connect(database = dbname, user = username)

# calculate the bounding box
latlon = np.array(map(float, bbox.split(','))).reshape(2,2).mean(axis=0)
latlon = latlon[1], latlon[0]
R = 6371.0  # earth radius in km
radius = 2.0  # km
x1 = latlon[1] - np.rad2deg(radius/R/np.cos(np.deg2rad(latlon[0])))
y1 = latlon[0] - np.rad2deg(radius/R)
x2 = latlon[1] + np.rad2deg(radius/R/np.cos(np.deg2rad(latlon[0])))
y2 = latlon[0] + np.rad2deg(radius/R)
# southwest, northeast
sbox = [x1, y1, x2, y2]

sql_query = """
SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
FROM photo_data_table
WHERE latitude > {lat_min} AND latitude < {lat_max} 
AND longitude > {lon_min} AND longitude < {lon_max};
""".format(lat_min=sbox[1], lat_max=sbox[3], lon_min=sbox[0], lon_max=sbox[2])


# sql_query = """
# SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
# FROM photo_data_table
# """

query_results = pd.read_sql_query(sql_query,con)
print query_results.shape

# remove duplicates
# sql_query = """
# SELECT DISTINCT * FROM photo_data_table
# """
# photo_data_from_sql = pd.read_sql_query(sql_query,con)
# photo_data_from_sql.to_sql('photo_data_table', engine, if_exists='replace')


hour_counts = []
for i in range(24):

    sql_query = """
    SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
    FROM photo_data_table
    WHERE DATE_PART('hour', datetaken) = {hour};
    """.format(hour=i)
    photo_data_from_sql = pd.read_sql_query(sql_query,con)
    print 'hour: ', i, photo_data_from_sql.shape[0], 'hits'

    hour_counts.append(photo_data_from_sql.shape[0])


plt.figure(figsize=(5,5))
plt.plot(hour_counts, '-o')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.xticks(np.arange(0,25,3),
           ['0AM', '3AM', '6AM', '9AM', '12PM', '3PM', '6PM', '9PM', '12AM'])
plt.xlim([0, 23])
plt.savefig('hour.png')
plt.show()

# sunday is 0
dow_counts = []
for i in range(7):

    sql_query = """
    SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
    FROM photo_data_table
    WHERE DATE_PART('dow', datetaken) = {dow};
    """.format(dow=i)
    photo_data_from_sql = pd.read_sql_query(sql_query,con)
    print 'dow: ', i, photo_data_from_sql.shape[0], 'hits'

    dow_counts.append(photo_data_from_sql.shape[0])


photo_data_from_sql.set_index('id').to_csv('results.csv', encoding='utf-8')

plt.figure(figsize=(5,5))
plt.plot(dow_counts, '-o', c=sb.color_palette()[1])
plt.xticks(range(7), ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat'], fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.savefig('dow.png')
plt.show()


# sunday is 0
day_counts = []
photo_data_from_sql = []
for i in range(1, 31):

    sql_query = """
    SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
    FROM photo_data_table

    WHERE datetaken >= '{day}'::date
    AND datetaken < ('{day}'::date + '1 day'::interval);

    """.format(day='09-%02d-11' %i)
    photo_data_from_sql.append(pd.read_sql_query(sql_query,con))
    # print 'hour: ', i, photo_data_from_sql.shape[0], 'hits'

    day_counts.append(photo_data_from_sql[i-1].shape[0])

plt.plot(day_counts, '-o')
plt.show()

# pd.concat(photo_data_from_sql).set_index('id').to_csv('results.csv', encoding='utf-8')


hour_counts = []
for i in range(24):
    for j in range(7):
    
        sql_query = """
        SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
        FROM photo_data_table
        WHERE DATE_PART('hour', datetaken) = {hour} 
        AND DATE_PART('dow', datetaken) = {dow};
        """.format(hour=i, dow=j)
        photo_data_from_sql = pd.read_sql_query(sql_query,con)
        print 'hour: ', i, photo_data_from_sql.shape[0], 'hits'

        hour_counts.append(photo_data_from_sql.shape[0])


plt.plot(hour_counts, '-o')
plt.show()



# clustering

# sql_query = """
# SELECT DISTINCT *
# FROM photo_data_table;
# """
# photo_data_from_sql = pd.read_sql_query(sql_query, con)
# print photo_data_from_sql.shape[0], 'hits'
# photo_data_from_sql.set_index('id', inplace=True)


# from sklearn.cluster import KMeans

# y_pred = KMeans(n_clusters=20, random_state=0).fit_predict(photo_data_from_sql[['latitude', 'longitude']])

# photo_data_from_sql['label'] = y_pred
# photo_data_from_sql.to_csv('results_labeled.csv', encoding='utf-8')
