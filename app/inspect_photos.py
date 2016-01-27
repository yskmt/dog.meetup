"""Example code of evaluating a Caffe reference model for ILSVRC2012 task.

Prerequisite: To run this example, crop the center of ILSVRC2012 validation
images and scale them to 256x256, and make a list of space-separated CSV each
column of which contains a full path to an image at the fist column and a zero-
origin label at the second column (this format is same as that used by Caffe's
ImageDataLayer).

"""
from __future__ import print_function
import argparse
import os
import sys
import random

import cv2
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions import caffe

import pandas as pd

model = 'bvlc_googlenet.caffemodel'
do_db = False
download = False
gpu = -1

###############################################################################
# read url list
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

if do_db:

    dbname = 'photo_db'
    username = 'ysakamoto'

    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))
    print(engine.url)

    con = None
    con = psycopg2.connect(database = dbname, user = username)

    # Generate a background image for the title page (just list of addresses)
    # get photos with most views
    sql_query = """
    SELECT DISTINCT *
    FROM photo_data_table
    ORDER BY views LIMIT 1000;
    """
    photo_popular = pd.read_sql_query(sql_query,con)
    photo_popular.to_csv('dog_photo_urls.csv', index=0)

###############################################################################
# download files
if download:

    import urllib

    testfile = urllib.URLopener()

    for i, idx in enumerate(photo_popular.index):
        photo = photo_popular.loc[idx]
        if i%10==0:
            print(i)
        testfile.retrieve(photo.url_m, 'photos/%d.%s'
                          %(photo.id,
                            photo.url_m.split('.')[-1]))

    
###############################################################################
# initialization

print('Loading Caffe model file %s...' % model, file=sys.stderr)
try:
    func
except NameError:
    func = caffe.CaffeFunction(model)
print('Loaded', file=sys.stderr)


in_size = 224
# Constant mean over spatial pixels
mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
mean_image[0] = 104
mean_image[1] = 117
mean_image[2] = 123

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


cropwidth = 256 - in_size
start = cropwidth // 2
stop = start + in_size
mean_image = mean_image[:, start:stop, start:stop].copy()
target_shape = (256, 256)
output_side_length=256


###############################################################################
# run CNN
# with open('synset_words.txt') as f:
#     with open('labels.txt', 'w') as fw:
#         for l in f:
#             fw.write(l[10:])

photo_popular = pd.read_csv('dog_photo_urls.csv', index_col=0)
categories = np.loadtxt("labels_dog.txt", str, delimiter="\t")
categories_dog = [i for i in range(len(categories)) if 'dog' in categories[i]][:-2]


def cnn_dog(photo_file, idx, categories_dog, gpu=-1, verbose=False):
        
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

                
    # pd.DataFrame(name_score, columns=['label', 'prob'])\
    #   .to_csv('photos/%d.csv' %(idx))


# dog_label = []
# for i, idx in enumerate(photo_popular.index[:10]):
#     photo = photo_popular.loc[idx]
    
#     if i%10==0:
#         print(i)
        
#     photo_file = 'photos/%d.%s' %(idx, photo.url_m.split('.')[-1])

#     dog_label.append(cnn_dog(photo_file, idx, categories_dog))




###############################################################################
# # postprocessing
# scores = []
# for i, idx in enumerate(photo_popular.index):
#     scores.append(np.loadtxt('photos/%d.csv' %(idx)))
# scores = np.array(scores).T


# dogs = []
# not_dogs = []
# for i, idx in enumerate(photo_popular.index):
#     nd = 1
#     for lb in np.argsort(scores[:,i])[::-1][:20]:
        
#         if lb in categories_dog:
#             dogs.append(idx)
#             nd = 0
#             break

#     if nd:
#         not_dogs.append(idx)

from PIL import Image

img_name = 'test/5862419122.jpg'

cnn_dog(img_name, 0, categories_dog, verbose=True)
image = Image.open(img_name)
image.show()


# for nd in not_dogs:
#     image = Image.open('photos/%d.jpg' %nd)
#     image.show()

