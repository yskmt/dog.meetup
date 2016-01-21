from flask import render_template
from app import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

from datetime import datetime

from geopy.geocoders import Nominatim, GoogleV3 

from app.cluster_photos import latlon_to_dist, get_bbox

from utils import get_ellipse_coords


# user = 'Katie' #add your username here (same as previous postgreSQL)
# host = 'localhost'
# dbname = 'birth_db'
# db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = user)

default_address = '260 Sheridan Ave, Palo Alto, CA 94306'

username = 'ysakamoto'
hostname = 'localhost'
dbname = 'photo_db'

engine = create_engine('postgres://%s@%s/%s'
                       %(username,hostname,dbname))
print engine.url

# connect:
con = None
con = psycopg2.connect(database = dbname, user = username)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
                           title = 'Home', user = { 'nickname': 'Miguel' },
    )

@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/map', methods=['GET'])
def map_output():
    query_address = request.args.get('address')
    query_time = int(request.args.get('time'))

    try:
        query_distance = float(request.args.get('distance'))
    except ValueError:
        query_distance = 5.0
        
    try:
        query_latlon = float(request.args.get('lat')), float(request.args.get('lon'))
    except:
        query_latlon = None
                
    if query_address == '':
        query_address = default_address

    if query_latlon is None:
        # try openmap geocoder first
        try:
            geolocator = Nominatim()
            location = geolocator.geocode(query_address)
        except:
            geolocator = GoogleV3(api_key='AIzaSyB7LvwvLJN0l04rFfHbIyUBsqi61vP6qWA')
            location = geolocator.geocode(query_address)
        query_latlon = (location.latitude, location.longitude)
            
    print 'latlon = ', query_latlon
    print 'time = ', query_time
    print 'address = ', query_address
    print 'distance = ', query_distance

    sbox = get_bbox(query_latlon, query_distance)

#     sql_query = """
# SELECT DISTINCT id,latitude,longitude,datetaken,description,tags,url_t 
# FROM photo_data_table
# WHERE latitude > {lat_min} AND latitude < {lat_max} 
# AND longitude > {lon_min} AND longitude < {lon_max}
# AND tags LIKE '%{tag}%'
# AND DATE_PART('hour', datetaken) = {hour};
# """.format(lat_min=sbox[1], lat_max=sbox[3],
#            lon_min=sbox[0], lon_max=sbox[2],
#            tag='dog', hour=query_time)

    sql_query = """
SELECT DISTINCT id,latitude,longitude,datetaken,description,tags,url_t,url_m
FROM photo_data_table
WHERE latitude > {lat_min} AND latitude < {lat_max} 
AND longitude > {lon_min} AND longitude < {lon_max};
""".format(lat_min=sbox[1], lat_max=sbox[3],
           lon_min=sbox[0], lon_max=sbox[2],
           tag='dog')

    query_results = pd.read_sql_query(sql_query, con)

    # convert latlon to xy coordinate in km
    xy = query_results[['latitude', 'longitude']]\
         .apply(lambda x: latlon_to_dist(x, query_latlon), axis=1)
         # .apply(lambda x: (x[0], x[1]), axis=1)
    xy = pd.DataFrame(xy, columns=['xy'])   
    for n, col in enumerate(['x', 'y']):
        xy[col] = xy['xy'].apply(lambda location: location[n])
       
    # convert datetaken to hour taken
    # scale: 1.0 means that 1 hour corresponds to 1 km
    scale = 10.0
    hours = query_results['datetaken'].apply(lambda x: x.hour+x.minute/60.0) * scale
    xyh = pd.concat([xy[['x', 'y']], hours], axis=1)

    # no photos around the center
    if xy[['x','y']].shape[0] == 0:
        return render_template("map.html",
                               photos=[{}],
                               num_labels=1,
                               address=query_address,
                               time=query_time,
                               distance=query_distance,
                               center=query_latlon)
    
    # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    labels = DBSCAN(eps=0.3, metric='euclidean', min_samples=5,
                    random_state=0)\
                    .fit_predict(xyh)
                    # .fit_predict(xy[['x','y']])

    # kde = KernelDensity(bandwidth=0.01,
    #                     kernel='exponential', algorithm='ball_tree')

    
    kde.fit(xyh)

    resolution = 100
    lats = np.linspace(query_results['latitude'].min(),
                       query_results['latitude'].max(), resolution)
    lons = np.linspace(query_results['longitude'].min(),
                       query_results['longitude'].max(), resolution)
    latsv, lonsv = np.meshgrid(lats, lons)
    
    samples = pd.DataFrame(np.array([latsv.flatten(),
                                     lonsv.flatten(),
                                     np.ones(resolution**2)*query_time]).T,
                           columns=['latitude', 'longitude', 'hour'])
    
    xy_ = samples[['latitude', 'longitude']].apply(lambda x: latlon_to_dist(x, query_latlon), axis=1)

    xy_ = pd.DataFrame(xy_, columns=['xy'])   
    for n, col in enumerate(['x', 'y']):
        xy_[col] = xy_['xy'].apply(lambda location: location[n])

    samples = pd.concat([samples, xy_[['x', 'y']]], axis=1)
    
    kde_score = np.exp(kde.score_samples(samples[['x','y','hour']]))
    kde_score /= np.max(kde_score)
    print kde_score

    nes = pd.concat([samples, pd.DataFrame(kde_score, columns=['proba'])], axis=1)
    
    # return only for the specified hour
    hours = query_results['datetaken'].apply(lambda x: x.hour)
    nes = nes[hours==query_time]

    
    # silave = []
    # for nc in range(2, 40, 4):
    #     print nc
    #     labels = KMeans(n_clusters=nc,
    #                     random_state=0)\
    #                     .fit_predict(xy[['x','y']])

    #     silave.append(silhouette_score(xy[['x','y']], labels))

    # nc = range(2, 40, 4)[np.argmax(silave)]
    # labels = KMeans(n_clusters=nc,
    #                 random_state=0)\
    #                 .fit_predict(xy[['x','y']])
                        
    # add labels to dataframe
    query_results = pd.concat([query_results,
                               pd.DataFrame(labels, columns=['label'])],
                              axis=1)

    # drop -1 clusters
    query_results = query_results[query_results['label']!=-1]

    # return only for the specified hour
    hours = query_results['datetaken'].apply(lambda x: x.hour)
    query_results = query_results[hours==query_time]

    # drop 1-element cluster after sliced by an hour
    min_cluster = 3
    idx_preserve = (query_results.groupby('label')['label'].count()!=min_cluster)
    idx_preserve = idx_preserve[idx_preserve==True]
    query_results = query_results[query_results.label.isin(idx_preserve.index)]
            
    # gather cluster characteristics
    lb_unique, num_pics = np.unique(labels, return_counts=True)
    num_pics = dict(zip(lb_unique, num_pics))
    centroids = query_results.groupby('label').mean().transpose().to_dict()
    for key, value in centroids.iteritems():
        value['num_pics'] = np.sqrt(num_pics[key])

    # get mean and covariance of the groups
    covs = query_results.groupby('label')[['latitude','longitude']].cov()
    means = query_results.groupby('label')[['latitude','longitude']].mean()
    num_pics = query_results.groupby('label')[['label']].count()
    num_pics.columns = ['num_pics']
    
    labels_multi = covs.index.get_level_values('label').unique()

    cluster_shape = {}
    for lb in labels_multi:
        eigs = np.linalg.eigh(covs.loc[lb])
        radii = list(np.sqrt(eigs[0])*2)
        pvec = eigs[1][:,0]  # direction of the 1st eigenvector (lat, lng)
        pvec = np.array([pvec[1], pvec[0]])  # switch to make it (lng, lat)
        pdir = [np.arctan(pvec[1]/pvec[0])]  # get angle from x-axis (lng-direction)
        center = list(means.loc[lb])

        # distance of the radii of 95% confident ellipse
        # print latlon_to_dist(np.array(center)+np.array(radii), center)
        
        cluster_shape[lb] = center + radii + pdir

    cluster_shape = pd.DataFrame(
        cluster_shape, index=['lat_c','lon_c','radii_x','radii_y','orientation'])

    cluster_shape = pd.concat([cluster_shape, num_pics.transpose()])    

    
    print '# clusters to show:', len(set(query_results['label']))
    
    return render_template("map.html",
                           photos=query_results.to_dict(orient='index'),
                           num_labels=len(set(query_results['label'])),
                           max_label=query_results['label'].max(),
                           address=query_address,
                           hour=datetime.strptime(str(query_time), "%H").strftime("%-I %p"),
                           distance=query_distance,
                           clusters=centroids,
                           cluster_shape=cluster_shape.to_dict(),
                           kde=nes.to_dict(orient='index'),
                           center=query_latlon)
