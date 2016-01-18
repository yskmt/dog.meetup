from flask import render_template
from app import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request

import numpy as np
from sklearn.cluster import DBSCAN

from app.cluster_photos import latlon_to_dist


# user = 'Katie' #add your username here (same as previous postgreSQL)
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

username = 'ysakamoto'
hostname = 'localhost'
dbname = 'photo_db'

engine = create_engine('postgres://%s@%s/%s'
                       %(username,hostname,dbname))
print engine.url

# connect:
con = None
con = psycopg2.connect(database = dbname, user = username)

# calculate the bounding box
bbox = '-122.511080,37.712002,-122.381984,37.809776'
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

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Miguel' },
       )

@app.route('/db')
def birth_page():
   
   sql_query = """
SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
FROM photo_data_table
WHERE latitude > {lat_min} AND latitude < {lat_max} 
AND longitude > {lon_min} AND longitude < {lon_max};
""".format(lat_min=sbox[1], lat_max=sbox[3], lon_min=sbox[0], lon_max=sbox[2])
   
   query_results = pd.read_sql_query(sql_query,con)
   urls = ""
   print query_results[:10]
   for i in range(0,10):
      urls += query_results.iloc[i]['url_t']
      urls += "<br>"
   return urls

@app.route('/db_fancy')
def cesareans_page_fancy():
   sql_query = """
SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
FROM photo_data_table
WHERE latitude > {lat_min} AND latitude < {lat_max} 
AND longitude > {lon_min} AND longitude < {lon_max};
""".format(lat_min=sbox[1], lat_max=sbox[3], lon_min=sbox[0], lon_max=sbox[2])
   
   query_results = pd.read_sql_query(sql_query,con)
   photos = []
   for i in range(0,query_results.shape[0]):
      photos.append(dict(id=query_results.iloc[i]['id'],
                       description=query_results.iloc[i]['description'],
                       url_t=query_results.iloc[i]['url_t']))
      
   return render_template('cesareans.html', photos=photos)

@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/output')
def cesareans_output():
   query_tag = request.args.get('tag')

   sql_query = """
SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
FROM photo_data_table
WHERE latitude > {lat_min} AND latitude < {lat_max} 
AND longitude > {lon_min} AND longitude < {lon_max}
AND tags LIKE '%{tag}%';
""".format(lat_min=sbox[1], lat_max=sbox[3], lon_min=sbox[0], lon_max=sbox[2],
           tag=query_tag)
   print sql_query
   
   query_results = pd.read_sql_query(sql_query, con)
   print query_results

   photos = []
   for i in range(0,query_results.shape[0]):
      photos.append(dict(id=query_results.iloc[i]['id'],
                         description=query_results.iloc[i]['description'],
                         url_t=query_results.iloc[i]['url_t'],
                         latitude=query_results.iloc[i]['latitude'],
                         longitude=query_results.iloc[i]['longitude'],
                         tags=query_results.iloc[i]['tags'],))
   the_result = ''
      
   return render_template("output.html", photos=photos, the_result=the_result)


@app.route('/map')
def map_output():
   query_tag = request.args.get('tag')

   sql_query = """
SELECT DISTINCT id,latitude,longitude,datetaken,description,tags,url_t 
FROM photo_data_table
WHERE latitude > {lat_min} AND latitude < {lat_max} 
AND longitude > {lon_min} AND longitude < {lon_max}
AND tags LIKE '%{tag}%'
AND DATE_PART('hour', datetaken) = {hour};
""".format(lat_min=sbox[1], lat_max=sbox[3], lon_min=sbox[0], lon_max=sbox[2],
           tag=query_tag, hour=15)
   
   query_results = pd.read_sql_query(sql_query, con)
   the_result = ''

   # convert latlon to xy coordinate in km
   xy = query_results[['latitude', 'longitude']]\
        .apply(lambda x: latlon_to_dist(x, latlon), axis=1)
   xy = pd.DataFrame(xy, columns=['xy'])   
   for n, col in enumerate(['x', 'y']):
       xy[col] = xy['xy'].apply(lambda location: location[n])
   
   # convert datetaken to hour taken
   # scale: 1.0 means that 1 hour corresponds to 1 km
   scale = 1.0
   hours = query_results['datetaken'].apply(lambda x: x.hour) * scale

   xyh = pd.concat([xy[['x', 'y']], hours], axis=1)

   # -1 for noisy samples
   labels = DBSCAN(eps=0.1, metric='euclidean', random_state=0)\
            .fit_predict(xyh)

   # add labels to dataframe
   query_results = pd.concat([query_results,
                              pd.DataFrame(labels, columns=['label'])],
                             axis=1)

   # drop -1s
   query_results = query_results[query_results['label']!=-1]
   
   return render_template("map.html",
                          photos=query_results.to_dict(orient='index'),
                          the_result=the_result,
                          center=latlon)
