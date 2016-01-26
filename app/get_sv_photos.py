"""

Get photos in sillicon valley area


"""
import pdb

import flickrapi

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

import numpy as np

from auths import fl0, fl1

from get_photos import gen_bboxes, get_pics


api_key = fl0
api_secret = fl1

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

dlon = 0.13
dlat = 0.10

left_bottom = [-122.57, 37.19]
right_top = [-121.94, 37.95]
n_lon = 5
n_lat = 8


bboxes = gen_bboxes(left_bottom[0], left_bottom[1], dlon, dlat, n_lon, n_lat)


###############################################################################
# connect to database

dbname = 'photo_db'
username = 'ysakamoto'

engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
print engine.url

if not database_exists(engine.url):
    create_database(engine.url)
print(database_exists(engine.url))

###############################################################################
# download photo info


nyc = [-74.3, 40.5]
bb = [nyc[0], nyc[1]]
n_lon = 5
n_lat = 4

austin = [-97.95, 30.14, -97.64, 30.46]
bb = [austin[0], austin[1]]
n_lon = 3
n_lat = 4

# [-73.7, 40.9]
# San Jose
# bbox = '-122.093452,37.155326,-121.766609,37.455173'

# for yr in xrange(2000, 2016):
#     print 'year', yr
#     dfs = get_pics(yr, bbox)

#     if len(dfs)>0:
#         dfs.to_sql('photo_data_table', engine, if_exists='append')


bboxes = gen_bboxes(bb[0], bb[1], dlon, dlat, n_lon, n_lat)

ct = 0
ct2 = 0
for bb in bboxes:
    print ct, '/', len(bboxes)
    bbox = ','.join(map(str, bb))
    print bbox
    for yr in xrange(2000, 2016):
        ct2 += 1
        print 'year', yr
        dfs = get_pics(yr, bbox)

        if len(dfs)>0:
            dfs.to_sql('photo_data_table', engine, if_exists='append')

    ct+=1
###############################################################################
# remove duplicaets
# con = None
# con = psycopg2.connect(database=dbname, user=username)
# cur = con.cursor()

# cur.execute("SELECT COUNT(DISTINCT id) FROM photo_data_table;")
# print cur.fetchone()

# cur.execute("CREATE TABLE phoit AS SELECT DISTINCT * FROM photo_data_table;")
# cur.execute("DROP TABLE photo_data_table;")
# cur.execute("ALTER TABLE phoit RENAME TO photo_data_table;")

# cur.execute("SELECT COUNT(DISTINCT id) FROM photo_data_table;")
# print cur.fetchone()

# con.commit()
# cur.close()
# con.close()


# # remove duplicate on pandas
# import pandas as pd

# con = None
# con = psycopg2.connect(database=dbname, user=username)

# sql_query = """
# SELECT DISTINCT *
# FROM photo_data_table
# """
# query_results = pd.read_sql_query(sql_query, con)

# query_results = query_results.drop_duplicates('id')

# query_results.to_sql('phoit', engine, if_exists='replace')

# con.close()
