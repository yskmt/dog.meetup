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


def get_inout_pics(inout, bbox):
    """
    Get photos based on indoor=1 or outdoor=2
    """

    dfs = []
    # maximum 40 pages
    # per page can go up to 500
    # use bbox    
    for i in xrange(1, 41):
        print 'page: ', i

        photos = flickr.photos.search(content_type=1,
                                      geo_context=inout,
                                      extras='geo,tags,url_t,url_m,'\
                                      'description,owner_name,views,path_alias,'\
                                      'date_upload,date_taken,machine_tags',
                                      per_page='500', page=i)

        # check if there is any results
        if len(photos['photos']['photo'])==0:
            break

        df_photos = pd.DataFrame(photos['photos']['photo'])

        # convert id to int and set it to index
        df_photos['id'] = df_photos['id'].astype(int)
        df_photos.set_index('id', inplace=True)
        
        # convert to int
        # height_o, pathalias, url_o, width_o can be null (_o for original)
        df_photos[['accuracy','context',
                   'datetakengranularity','datetakenunknown',
                   'farm',
                   'geo_is_contact','geo_is_family',
                   'geo_is_friend','geo_is_public',
                   'height_t','height_m',
                   'isfamily', 'ispublic', 'isfriend',
                   'server','views',
                   'width_t','width_m',
                   'woeid']]\
                   = df_photos[['accuracy','context',
                                'datetakengranularity','datetakenunknown',
                                'farm',
                                'geo_is_contact','geo_is_family',
                                'geo_is_friend','geo_is_public',
                                'height_t','height_m',
                                'isfamily', 'ispublic', 'isfriend',
                                'server','views',
                                'width_t','width_m',
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
        dfs.to_csv('photos-inout-%d.csv' %inout, encoding='utf-8')

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

# indoor=1 or outdoor=2
# for i in xrange(1, 3):
#     print 'inout', i
#     dfs = get_inout_pics(i, bbox)

#     if len(dfs)>0:
#         dfs.to_sql('photo_inout_table', engine, if_exists='append')


###############################################################################
# connect:
con = None
con = psycopg2.connect(database=dbname, user=username)
cur = con.cursor()

cur.execute("SELECT COUNT(DISTINCT id) FROM photo_inout_table;")
print cur.fetchone()

# remove duplicates
cur.execute("CREATE TABLE phoit AS SELECT DISTINCT * FROM photo_inout_table;")
cur.execute("DROP TABLE photo_inout_table;")
cur.execute("ALTER TABLE phoit RENAME TO photo_inout_table;")

cur.execute("SELECT COUNT(DISTINCT id) FROM photo_inout_table;")
print cur.fetchone()

con.commit()
cur.close()
con.close()

# get features for indoor/outdoor
con = None
con = psycopg2.connect(database=dbname, user=username)

sql_query = """
SELECT * FROM photo_inout_table
WHERE context = 1;
"""
photo_indoor = pd.read_sql_query(sql_query,con)

sql_query = """
SELECT * FROM photo_inout_table
WHERE context = 2;
"""
photo_outside = pd.read_sql_query(sql_query,con)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold

tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1,
                                max_features=1000,
                                stop_words='english')


sql_query = """
SELECT * FROM photo_inout_table"""
photo_inout = pd.read_sql_query(sql_query, con)


# k-fold cross validation

photo_description = photo_inout.description.drop_duplicates()

kf = KFold(n=photo_description.shape[0], n_folds=8, shuffle=True, random_state=1)
tf = tf_vectorizer.fit_transform(photo_inout.description)

for train_index, test_index in kf:

    X_train = tf[train_index, :]
    X_test = tf[test_index, :]

    y_train = photo_inout.context.iloc[train_index]
    y_test = photo_inout.context.iloc[test_index]

    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # confusion matrix
    confmat = confusion_matrix(y_test, y_pred)

    print confmat



# check out indoor/outoodr pictures
pio = photo_inout.drop_duplicates('description')

pio_out = pio[pio.context==2]

for url in pio_out.url_t[:20]:
    fd = urllib.urlopen(url)
    image_file = io.BytesIO(fd.read())
    im = Image.open(image_file)
    im.show()



###############################################################################
# test out

sql_query = """
SELECT * FROM photo_data_table WHERE tags LIKE '%dog%' ORDER BY random() LIMIT 100;
"""

photos = pd.read_sql_query(sql_query,con)

tf = tf_vectorizer.fit_transform(photo_inout.description)
tf_test = tf_vectorizer.transform(photos.description)

clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(tf, photo_inout.context)
preds = clf.predict(tf_test)

for i in range(photos.shape[0]):

    # indoor: 1, outdoor: 2
    if preds[i]==1:
        print i
        fd = urllib.urlopen(photos.iloc[i].url_t)
        image_file = io.BytesIO(fd.read())
        im = Image.open(image_file)
        im.show()

for pt in photos.tags:
    print pt
