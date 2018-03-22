# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 21:10:14 2018

@author: arbaaz
"""

from PIL import Image
from random import shuffle
import glob
import numpy as np
import tables
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
# import numexpr

hdf5_path = 'C:/Users/arbaaz/Desktop/Vision Project/isprs.h5'

IMAGE_PATH = "C:/Users/arbaaz/Desktop/Vision Project/Processed Data/datasets/potsdam/Train/*"
LABELS_PATH = "C:/Users/arbaaz/Desktop/Vision Project/Processed Data/datasets/potsdam/Labels/*"
#
#labels = IMAGE_PATH + 'Labels/*'
#data = IMAGE_PATH + 'Train/*'

hdf5_file = tables.open_file(hdf5_path, mode='w')

addrs = glob.glob(IMAGE_PATH)
labels = glob.glob(LABELS_PATH)
labels = [item.replace("Train", "Labels") for item in addrs];
data_len = [i for i, e in enumerate(addrs)]
j = len(data_len)
a = cv2.imread(addrs[1])

# Dividing the data into 75% train, 15% validation, and 10% test
train_addrs = addrs[0:int(0.85*j)]
train_labels = labels[0:int(0.85*j)]

val_addrs = addrs[int(0.85*j):int(1.0*j)]
val_labels = labels[int(0.85*j):int(1.0*j)]

img_dtype = tables.UInt8Atom()
data_shape = (0, 256, 256, 3)
# hdf5_file = tables.open_file(hdf5_path, mode='a')

folder = hdf5_file.create_group('/','DS')

train_storage = hdf5_file.create_earray(folder, 
                                        'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_earray(folder, 
                                      'val_img', img_dtype, shape=data_shape)
mean_storage = hdf5_file.create_earray(hdf5_file.root, 
                                      'train_mean', img_dtype, shape=data_shape)

train_lab = hdf5_file.create_earray(folder, 
                                        'train_labels', img_dtype, shape=data_shape)
val_lab = hdf5_file.create_earray(folder, 
                                      'val_labels', img_dtype, shape=data_shape)
mean = np.zeros(data_shape[1:], np.float32)

for i in range(len(train_addrs)):

    addr = train_addrs[i]
    img = cv2.imread(addr)
#    plt.imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    label = train_labels[i]
    img_tar = cv2.imread(label)
    img_tar = cv2.cvtColor(img_tar, cv2.COLOR_BGR2RGB)
    plt.imshow(img_tar)        

    train_storage.append(img[None])
    train_lab.append(img_tar[None])
    mean += img / float(len(train_labels))



# to shuffle data
# if shuffle_data:
#    c = list(zip(addrs, labels))
#    shuffle(c)
#    addrs, labels = zip(*c)

