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

import os
import flickrapi
import numpy as np
import urllib
import pandas as pd
from tqdm import tqdm

# from inspect_photos import cnn_dog
import chainer
import chainer.functions as F
from chainer.functions import caffe
import cv2

from auths import fl0, fl1

api_key = fl0
api_secret = fl1

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')

bbox = '-123.376688,28.437433,-62.115815,48.911746'


def get_pics(bbox):

    dfs = []
    # maximum 40 pages
    # per page can go up to 500
    # use bbox
    
    for i in xrange(1, 11):
        print 'page: ', i

        # geo_context -  0: not defined, 1: indoors, 2: outdoors
        photos = flickr.photos.search(bbox=bbox,
                                      content_type=1,
                                      text='dog',
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
        dfs.to_csv('random/photos.csv', encoding='utf-8')

    return dfs



def cnn_dog_proba(photo_file, idx, categories_dog, func, gpu=-1, verbose=False,
                  save_csv=True):


    in_size = 224

    # Constant mean over spatial pixels
    mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
    mean_image[0] = 104
    mean_image[1] = 117
    mean_image[2] = 123

    cropwidth = 256 - in_size
    start = cropwidth // 2
    stop = start + in_size
    mean_image = mean_image[:, start:stop, start:stop].copy()
    target_shape = (256, 256)
    output_side_length=256
    
    def forward(x, t):
        y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'],
                  train=False)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    def predict(x):
        y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
                  disable=['loss1/ave_pool', 'loss2/ave_pool'],
                  train=False)
        return F.softmax(y)
    
    image = cv2.imread(photo_file)
    height, width, depth = image.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
     new_height = output_side_length * height / width
    else:
     new_width = output_side_length * width / height
    resized_img = cv2.resize(image, (new_width, new_height))
    height_offset = (new_height - output_side_length) / 2
    width_offset = (new_width - output_side_length) / 2
    image= resized_img[height_offset:height_offset + output_side_length,
    width_offset:width_offset + output_side_length]

    image = image.transpose(2, 0, 1)
    image = image[:, start:stop, start:stop].astype(np.float32)
    image -= mean_image
    x_batch = np.ndarray(
            (1, 3, in_size,in_size), dtype=np.float32)
    x_batch[0]=image

    if gpu >= 0:
      x_batch=cuda.to_gpu(x_batch)
    x = chainer.Variable(x_batch, volatile=True)
    score = predict(x)

    if gpu >= 0:
      score=cuda.to_cpu(score.data)
    # print(score.data)

    if save_csv:
        np.savetxt('photos/%d.csv' %(idx), score.data[0])

    return np.sum(score.data[0][categories_dog])



###############################################################################
# Convert pandas to sql
# if __name__ == "__main__":
if 1:
    # dfs = get_pics(bbox)

    dfs = pd.read_csv('random/photos.csv', encoding='utf-8')
    
    # for i in xrange(dfs.shape[0]):

    #     photo = dfs.iloc[i]
    #     photo_url = photo.url_m
    #     if photo_url is None:
    #         photo_url = photo.url_t
        
    #     photo_name = photo_url, 'random/%d.%s' %(photo.id,
    #                                              photo_url.split('.')[-1])

    #     if not os.path.exists(photo_name[1]):
    #         urllib.urlretrieve(*photo_name)

    categories = np.loadtxt("labels_dog.txt", str, delimiter="\t")
    categories_dog = [i for i in range(len(categories)) if 'dog' in categories[i]][:-2]

    model = 'bvlc_googlenet.caffemodel'
    do_db = False
    download = False
    gpu = -1

    print 'Loading Caffe model file %s...' % model
    try:
        func
    except NameError:
        func = caffe.CaffeFunction(model)
    print 'Loaded'
    
    dog = {}
    for f in tqdm(os.listdir('random/dog')):
        idx = int(f.split('.')[0])
        dog[idx] = cnn_dog_proba('random/dog/'+f, idx, categories_dog, func)

    not_dog = {}
    for f in tqdm(os.listdir('random/not_dog')):
        if 'jpg' not in f:
            continue
        idx = int(f.split('.')[0])
        not_dog[idx] = cnn_dog_proba('random/not_dog/'+f, idx, categories_dog, func)

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


false_negative = len([dog[d] for d in dog if dog[d] is False])
false_positive = len([not_dog[d] for d in not_dog if not_dog[d] is True])

from sklearn.metrics import roc_curve, auc

y_score = dog.values() + not_dog.values()
y_test = [1] * len(dog) + [0]*len(not_dog)

fpr, tpr, threshold = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('Receiver operating characteristic', fontsize=15)
plt.legend(loc="lower right", fontsize=15)
plt.savefig('roc.png')
plt.show()

th_use = 0.85
print th_use
print threshold[tpr>th_use][0]
print fpr[tpr>th_use][0]


df_val = pd.DataFrame([y_score, y_test]).transpose()
df_val.columns = ['pred', 'test']
df_val.to_csv('random/validation.csv')


