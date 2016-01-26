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

from auths import fl0, fl1

api_key = fl0
api_secret = fl1

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

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

def gen_bboxes(lon0, lat0, dlon, dlat, n_lon, n_lat):
    """
    starting from very left-botton
    going down n_lat time, goint right n_lon times

    """

    bboxes = []
    for i in range(n_lat):
        for j in range(n_lon):
    
            bboxes.append(
                [lon0+dlon*j, lat0+dlat*i, lon0+dlon*(j+1), lat0+dlat*(i+1)])

    return bboxes


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
                                      extras='geo,tags,url_s,url_t,url_m,'\
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
        dfs.to_csv('photos-%d.csv' %year, encoding='utf-8')

    return dfs



###############################################################################
# Convert pandas to sql
if __name__ == "__main__":

    dbname = 'photo_db'
    username = 'ysakamoto'

    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    print engine.url

    if not database_exists(engine.url):
        create_database(engine.url)
    print(database_exists(engine.url))

    # # 2001 - 2016
    # for yr in xrange(2000, 2016):
    #     print 'year', yr
    #     dfs = get_pics(yr, bbox)

    #     if len(dfs)>0:
    #         dfs.to_sql('photo_data_table', engine, if_exists='append')


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


    # ###############################################################################
    # # remove duplicaets
    # con = None
    # con = psycopg2.connect(database=dbname, user=username)
    # cur = con.cursor()

    # cur.execute("SELECT COUNT(DISTINCT id) FROM photo_inout_table;")
    # print cur.fetchone()

    # # remove duplicates
    # cur.execute("CREATE TABLE phoit AS SELECT DISTINCT * FROM photo_inout_table;")
    # cur.execute("DROP TABLE photo_inout_table;")
    # cur.execute("ALTER TABLE phoit RENAME TO photo_inout_table;")

    # cur.execute("SELECT COUNT(DISTINCT id) FROM photo_inout_table;")
    # print cur.fetchone()

    # con.commit()
    # cur.close()
    # con.close()

    ###############################################################################
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
    plt.plot(hour_counts, '-o', linewidth=3)
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
    plt.plot(dow_counts, '-o', c=sb.color_palette()[1], linewidth=4)
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


    hour_counts = []
    for j in range(7):
        for i in range(24):

            sql_query = """
            SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
            FROM photo_data_table
            WHERE DATE_PART('hour', datetaken) = {hour} 
            AND DATE_PART('dow', datetaken) = {dow};
            """.format(hour=i, dow=j)
            photo_data_from_sql = pd.read_sql_query(sql_query,con)
            print 'hour: ', i, photo_data_from_sql.shape[0], 'hits'

            hour_counts.append(photo_data_from_sql.shape[0])

    plt.figure(figsize=(5,5))
    plt.plot(hour_counts, '-o', linewidth=2, c=sb.color_palette()[0])
    plt.xticks(np.arange(12,24*7+12,24),
               ['Sun', 'Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat'],fontsize=15)
    plt.xlim([0, 24*7])
    plt.tight_layout()
    plt.savefig('dow-hour.png')
    plt.show()

    month_counts = []
    for i in range(1,13):

        sql_query = """
        SELECT DISTINCT id,latitude,longitude,description,tags,url_t 
        FROM photo_data_table
        WHERE DATE_PART('month', datetaken) = {month} 
        """.format(month=i)
        photo_data_from_sql = pd.read_sql_query(sql_query,con)
        print 'month: ', i, photo_data_from_sql.shape[0], 'hits'

        month_counts.append(photo_data_from_sql.shape[0])

    plt.plot(range(1,13), month_counts, '-o')
    plt.show()


    # # Generate a background image for the title page (just list of addresses)
    # # get photos with most views
    # sql_query = """
    # SELECT DISTINCT *
    # FROM photo_data_table
    # ORDER BY views;
    # """
    # photo_popular = pd.read_sql_query(sql_query,con)

    # num = 15**2

    # popu = photo_popular.iloc[:num]['url_t']
    # ', '.join(map(lambda x: 'url(\''+x+'\')', popu.tolist()))
    # ', '.join(['no-repeat']*num)


    # popu = photo_popular.iloc[:num]['url_s']

    # for i in range(15):
    #     for j in range(15):
    #         print 'url('+ popu[i*10+j] +') ' + str(i*100) +'px '+ str(j*100) + 'px no-repeat,' 


    import Image

    #opens an image:
    im = Image.open("1_tree.jpg")
    #creates a new empty image, RGB mode, and size 400 by 400.
    new_im = Image.new('RGB', (400,400))

    #Here I resize my opened image, so it is no bigger than 100,100
    im.thumbnail((100,100))
    #Iterate through a 4 by 4 grid with 100 spacing, to place my image
    for i in xrange(0,500,100):
        for j in xrange(0,500,100):
            #I change brightness of the images, just to emphasise they are unique copies.
            im=Image.eval(im,lambda x: x+(i+j)/30)
            #paste the image at location i,j:
            new_im.paste(im, (i,j))

    new_im.show()
