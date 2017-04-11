#!/usr/bin/env python
# coding: utf8

# make sure that caffe is on the python path
CAFFE_ROOT = '/opt/caffe/caffe-master/'
import sys
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe

import os
import glob
import cv2
import caffe
import numpy as np
from caffe.proto import caffe_pb2


# AVA
AVA_ROOT = '/Datasets/AVA/'
IMAGE_MEAN= AVA_ROOT + 'mean_AADB_regression_warp256.binaryproto'
DEPLOY = AVA_ROOT + 'initModel.prototxt'
MODEL_FILE = AVA_ROOT + 'initModel.caffemodel'
IMAGE_FILE = AVA_ROOT + "*jpg"


#caffe.set_mode_gpu()
caffe.set_mode_cpu()

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

input_layer = 'imgLow'

'''
Image processing helper function
'''
def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

'''
Reading mean image, caffe model and its weights
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(IMAGE_MEAN) as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

#Read model architecture and trained model's weights
net = caffe.Net(DEPLOY,
                MODEL_FILE,
                caffe.TEST)

#Define image transformers
print "Shape mean_array : ", mean_array.shape
print "Shape net : ", net.blobs[input_layer].data.shape
net.blobs[input_layer].reshape(1,        # batch size
                              3,         # channel
                              IMAGE_WIDTH, IMAGE_HEIGHT)  # image size
transformer = caffe.io.Transformer({input_layer: net.blobs[input_layer].data.shape})
transformer.set_mean(input_layer, mean_array)
transformer.set_transpose(input_layer, (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob(IMAGE_FILE)]

#Making predictions
test_ids = []
preds = []
best_image = ''
best_score = 0.0
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net.blobs[input_layer].data[...] = transformer.preprocess(input_layer, img)
    out = net.forward()
    print out
    #{'fc9_ColorHarmony': array([[ 0.22087261]], dtype=float32), 'fc9_MotionBlur': array([[-0.08059776]], dtype=float32), 'fc9_Light': array([[ 0.14933866]], dtype=float32), 'fc9_Content': array([[ 0.01467544]], dtype=float32), 'fc9_Repetition': array([[ 0.18157494]], dtype=float32), 'fc11_score': array([[ 0.55613178]], dtype=float32), 'fc9_DoF': array([[-0.05279735]], dtype=float32), 'fc9_VividColor': array([[ 0.13607402]], dtype=float32), 'fc9_Symmetry': array([[ 0.06802807]], dtype=float32), 'fc9_Object': array([[ 0.00289625]], dtype=float32), 'fc9_BalancingElement': array([[-0.04946293]], dtype=float32), 'fc9_RuleOfThirds': array([[-0.0477073]], dtype=float32)}


    pred_score = out['fc11_score'][0][0]
    print img_path, '\t', pred_score
    if pred_score > best_score:
        #print "Better score !"
        best_score = pred_score
        best_image = img_path

    #test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    #preds = preds + [pred_probas.argmax()]


    #print pred_probas.argmax()
    #print '-------'
print "Best image, based only on fc11_score = ", best_image
