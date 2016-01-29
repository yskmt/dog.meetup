"""Example code of evaluating a Caffe reference model for ILSVRC2012 task.

Prerequisite: To run this example, crop the center of ILSVRC2012 validation
images and scale them to 256x256, and make a list of space-separated CSV each
column of which contains a full path to an image at the fist column and a zero-
origin label at the second column (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import sys
import os.path

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
import urllib
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe

import pandas as pd



def cnn_dog(photo_file, idx, categories_dog, func, gpu=-1, verbose=False,
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
    
    sd = np.argsort(score.data[0])[::-1]

    if verbose:
        top_k = 20
        prediction = zip(score.data[0].tolist(), categories)
        prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)

        # name_score = []
        for rank, (score, name) in enumerate(prediction[:top_k], start=1):
            print('#%d | %s | %4.1f%%' % (rank, name, score * 100))
            # name_score.append([name, score])
    
    # check if the top 20 labels have dog-relate ones
    top_k = 20
    for lb in sd[:top_k]:
        if lb in categories_dog:
            return True
            break

    return False


def get_run_cnn(dbname, username, offset,
                categories_dog, func, limit=100):

    # get photo data from database
    con = None
    con = psycopg2.connect(database=dbname, user=username)
    
    sql_query = """
    SELECT DISTINCT *
    FROM photo_data_table
    ORDER BY index LIMIT {limit} OFFSET {offset};
    """.format(limit=limit, offset=offset)
    photos = pd.read_sql_query(sql_query,con)

    con.close()

    # download file and run cnn
    for i in xrange(photos.shape[0]):

        photo = photos.loc[i]
        photo_url = photo.url_m
        if photo_url is None:
            photo_url = photo.url_t
        
        photo_name = photo_url, 'photos/%d.%s' %(photo.id,
                                                 photo_url.split('.')[-1])

        if not os.path.exists(photo_name[1]):
            urllib.urlretrieve(*photo_name)

        if not os.path.exists(photo_name[1].replace('jpg', 'csv')):
            cnn_dog(photo_name[1], photo.id, categories_dog, func)

    
###############################################################################
# initialization of the caffe model

# if len(sys.argv) < 3:
#     sys.exit()

# st = int(sys.argv[1])
# ed = int(sys.argv[2])

# print ('start', st)
# print ('end', ed)

categories = np.loadtxt("labels_dog.txt", str, delimiter="\t")
categories_dog = [i for i in range(len(categories)) if 'dog' in categories[i]][:-2]

# model = 'bvlc_googlenet.caffemodel'
# do_db = False
# download = False
# gpu = -1

# print('Loading Caffe model file %s...' % model, file=sys.stderr)
# try:
#     func
# except NameError:
#     func = caffe.CaffeFunction(model)
# print('Loaded', file=sys.stderr)


###############################################################################
# run cnn

# dbname = 'photo_db'
# username = 'ysakamoto'

# limit = 10

# # get the total number of photos
# con = None
# con = psycopg2.connect(database=dbname, user=username)
# sql_query = """
# SELECT COUNT(id)
# FROM photo_data_table
# """
# n_photos = pd.read_sql_query(sql_query,con).values[0][0]
# con.close()

# for i in tqdm(xrange(st, min(ed, n_photos/limit+1))):
#     get_run_cnn(dbname, username, i*limit, categories_dog, func, limit)


###############################################################################
# postprocessing

dog = {}
top_k = 20
ct = 0

for f in tqdm(os.listdir('photos')):
    if ct >100:
        break
    ct+=1
    
    if 'csv' not in f:
        continue
    
    id = f.split('.')[0]
    dog[id] = (np.sum(np.loadtxt('photos/'+f)[categories_dog]))

df_dog = pd.DataFrame(dog.values(), index=dog.keys(), columns=['dog_proba'])

# save the dog scores to database

dbname = 'photo_db'
username = 'ubuntu'

engine = create_engine('postgres://%s@/%s'%(username,dbname))
print(engine.url)

df_dog.to_sql('dog_proba_table', engine, if_exists='append')
        
# from PIL import Image

# img_name = 'test/5862419122.jpg'

# cnn_dog(img_name, 0, categories_dog, verbose=True)
# image = Image.open(img_name)
# image.show()


# # for nd in not_dogs:
# #     image = Image.open('photos/%d.jpg' %nd)
# #     image.show()

