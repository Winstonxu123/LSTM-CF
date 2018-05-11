#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:30:56 2017

@author: xu
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb



def img_middle_crop(image):
    start_h = np.uint8((image.shape[0] - 426) / 2)
    start_w = np.uint8((image.shape[1] - 426) / 2)
    end_h = start_h + 426
    end_w = start_w + 426
    croped_img = image[start_h:end_h, start_w:end_w,:]
    return croped_img



openString='img_0063.jpg'
oringal='images/'+openString
srcImg=cv2.imread(oringal)
ImgAfterCrop=img_middle_crop(srcImg)
segments1=felzenszwalb(ImgAfterCrop,scale=500,sigma=0.5,min_size=50)

plt.subplot(1,2,1)
plt.imshow(ImgAfterCrop)
plt.subplot(1,2,2)
plt.imshow(segments1)

saveString='Seg'+openString
cv2.imwrite(saveString,segments1)

img=cv2.imread(saveString)
cv2.imwrite(saveString,img)



