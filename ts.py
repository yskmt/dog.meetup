"""
https://www.flickr.com/services/api/

flickr API limit: under 3600 queries per hour

"""
import pdb

import flickrapi

from PIL import Image
import urllib2 as urllib
import io

import urllib, cStringIO
import PIL.Image as Image

import pandas as pd


api_key = u'264d5eb00e8ca0a446d714cad1e4a99a'
api_secret = u'79f9b05a9003ace7'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

# photos = flickr.photos.search(user_id='44247215@N00', per_page='10')
# sets = flickr.photosets.getList(user_id='44247215@N00')

lat = '37.788554'
lon = '-122.408083'

dfs = []
for i in xrange(11, 101):
    print 'page: ', i
    
    photos = flickr.photos.search(lat=lat, lon=lon, accuracy=15, #radius=2,
                                  content_type=1,
                                  extras='geo,tags,url_t,url_m,description,'\
                                  'date_upload,date_taken,machine_tags',
                                  per_page='100', page=i)
    
    df_photos = pd.DataFrame(photos['photos']['photo'])
    df_photos.set_index('id', inplace=True)

    # just take the content of the dictionary
    df_photos['description'] \
        = df_photos['description'].apply(lambda x: x['_content'])

    # convert to datetime format
    df_photos['datetaken'] = pd.to_datetime(df_photos['datetaken'])
    df_photos['dateupload'] \
        = pd.to_datetime(df_photos['dateupload'].astype(int), unit='s')
    
    dfs.append(df_photos)

dfs = pd.concat(dfs)

dfs.to_csv('photos.csv', encoding='utf-8')

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



###############################################################################
# Convert pandas to sql

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

dbname = 'photo_db'
username = 'ysakamoto'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine.url

if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))

dfs.to_sql('photo_data_table', engine, if_exists='append')


# connect:
con = None
con = psycopg2.connect(database = dbname, user = username)

keyword = "restaurant"

sql_query = """
SELECT id,latitude,longitude,description,tags,url_t FROM photo_data_table WHERE tags LIKE '%{query}%';
""".format(query=keyword)

photo_data_from_sql = pd.read_sql_query(sql_query,con)

print photo_data_from_sql.shape[0], 'hits'


photo_data_from_sql.set_index('id').to_csv('results.csv', encoding='utf-8')
