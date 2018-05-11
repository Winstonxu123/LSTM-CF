#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 04:31:48 2017

@author: xu
"""
caffe_root = '/home/xu/myWorkspace/LSTM/'  
import sys
sys.path.insert(0, caffe_root + 'python')


import caffe
caffe.set_mode_cpu()
#net = caffe.Net('../SUNRGBD/deploy.prototxt', '../SUNRGBD/SUNRGBD_final.caffemodel', caffe.TEST)
def img_middle_crop(image):
    start_h = np.uint8((image.shape[0] - 426) / 2)
    start_w = np.uint8((image.shape[1] - 426) / 2)
    end_h = start_h + 426
    end_w = start_w + 426
    croped_img = image[start_h:end_h, start_w:end_w]
    return croped_img

import matplotlib.pyplot as plt
import numpy as np
image_name = 'images/img_0063.jpg'
gt_name = 'gt/img_0063.jpg'
hha_name = 'hha/img_0063_abs.png'

image_blob=caffe.io.load_image(hha_name,color=False)
cropimg=img_middle_crop(image_blob)
cropimg = np.transpose(cropimg, [2, 0, 1])
print np.shape(image_blob)
image_blob=np.squeeze(image_blob)
image_blob=image_blob*(-1)
image_blob=np.abs(image_blob)
#imageMat=plt.imread(image_blob)
hha=plt.imread(hha_name)
print np.shape(hha)
plt.imshow(image_blob)
#plt.show()


#net.blobs['data'].reshape(1,1,426,426)
#net.blobs['data'].data[...] = image_blob

    
    # crop image

    
    #image_blob = image_blob[:,:,::-1]                 # convert from RGB to BGR
    
    # subtract the mean of BGR
    #image_blob -= np.array((110.324,116.435,125.793))
    #image_blob -= 110.324
     # permute width and height
    
    #hha_blob = hha_blob[:, :, ::-1]                  # convert from HHA to AHH
    # subtract the mean of AHH
    #hha_blob -= np.array((115.042,231.247,20.523))
    #hha_blob -= 115.042
