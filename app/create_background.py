import psycopg2
import pandas as pd
import numpy as np
import os
import urllib
from PIL import Image, ImageOps


dbname = 'aws_db'
username = 'ysakamoto'

con = None
con = psycopg2.connect(database = dbname, user = username)


# generate background
sql_query = """
SELECT DISTINCT photo_data_table.id,latitude,longitude,datetaken,url_m,
description,tags,url_t,dog_proba,views
FROM dog_proba_table 
INNER JOIN photo_data_table 
ON (dog_proba_table.index = photo_data_table.id)
ORDER BY views
"""
query_results = pd.read_sql_query(sql_query, con)
# dog_proba = query_results[map(str, categories_dog)].sum(axis=1)

# filter non-dogs
query_results = query_results[query_results['dog_proba']>0.85]

num = 800
popu = query_results.iloc[:num]['url_m'].values
np.random.shuffle(popu)


ims = 100
new_im = Image.new('RGB', (13*ims,7*ims))

for i in xrange(13):
    for j in xrange(7):
        print i,j

        photo_name = popu[i*10+j], 'background/%s' %(popu[i*10+j].split('/')[-1])

        # download image
        if not os.path.exists(photo_name[1]):
            urllib.urlretrieve(*photo_name)

        # open image
        im = Image.open(photo_name[1])
        im = ImageOps.fit(im, (100,100), Image.ANTIALIAS)

        if (i>2) and (i<10):
            im = np.ones(np.array(im).shape)*0
            im = Image.fromarray(im.astype(np.uint8))
            
        #paste the image at location:
        new_im.paste(im, (i*ims+j,j*ims))

new_im.show()
new_im.save('background.jpg')

import matplotlib.pyplot as plt
plt.imshow(new_im, cmap=plt.cm.Greys_r)
plt.show()
