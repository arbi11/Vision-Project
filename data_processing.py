# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 22:41:37 2018

@author: arbaaz
"""

import numpy as np
from skimage import io
from glob import glob
# from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
import time
import scipy.misc


# Parameters
WINDOW_SIZE = (256, 256) # Patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
FOLDER = "C:/Users/arbaaz/Desktop/Vision Project/Processed Data/datasets/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10 # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS) # Number of classes
# WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
CACHE = True # Store the dataset in-memory

DATASET = 'Potsdam'

if DATASET == 'Potsdam':
    MAIN_FOLDER = FOLDER + 'potsdam/'
    DATA_FOLDER = MAIN_FOLDER + '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
    PRO_DATA = MAIN_FOLDER + 'Processed Data/'    
elif DATASET == 'Vaihingen':
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
    
# ISPRS color palette
# Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

# We load one tile from the dataset and we display it
#img = io.imread('./top_potsdam_2_10_RGBIR.tif')
#fig = plt.figure()
#fig.add_subplot(121)
#plt.imshow(img)
#
## We load the ground truth
#gt = io.imread('./top_potsdam_2_10_label.tif')
#fig.add_subplot(122)
#plt.imshow(gt)
#plt.show()
#
## We also check that we can convert the ground truth into an array format
#array_gt = convert_from_color(gt)
#print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)

# Load the datasets
#if DATASET == 'Potsdam':
#    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
#    all_ids = ["_".join(f.split('_')[3:5]) for f in all_files]
#elif DATASET == 'Vaihingen':
#    #all_ids = 
#    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
#    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
## Random tile numbers for train/test split
#train_ids = random.sample(all_ids, 2 * len(all_ids) // 3 + 1)
#test_ids = list(set(all_ids) - set(train_ids))

# Exemple of a train/test split on Vaihingen :
train_ids = ['2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11', '4_12', 
              '5_10', '5_11', '5_12', '6_7', '6_8', '6_9', '6_10', '6_11', '6_12', 
              '7_7', '7_8', '7_9', '7_10', '7_11', '7_12']

test_ids = ['4_12', '5_13','2_13','5_14', '5_15', '6_13', '6_14', '6_15', '7_13', '2_14','3_13', '3_14','4_13', '4_14',] 

print("Tiles for training : ", train_ids)
print("Tiles for testing : ", test_ids)

data_files = [DATA_FOLDER.format(id) for id in train_ids]
label_files = [LABEL_FOLDER.format(id) for id in train_ids]

# label = np.asarray(convert_from_color(io.imread(label_files[random_idx])), dtype='int64')

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[0:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


window_shape = [256, 256]

for i in range(500):
    
    random_idx = random.randint(0, len(data_files) - 1)
    img = io.imread(data_files[random_idx])
    label = io.imread(label_files[random_idx])

    x1, x2, y1, y2 = get_random_pos(img, window_shape)

# Wait for 5 seconds
    time.sleep(0.5)
    data_p = img[x1:x2,y1:y2, :]
    label_p = label[x1:x2,y1:y2, :]

    attr = train_ids[random_idx] + '_' + str(x1) + '_' + str(y1)
    fileImg = MAIN_FOLDER + 'Train/potsdam_' + attr + '.png'
    
# Wait for 5 seconds
    time.sleep(0.5)
    data_f = data_p[:,:,0:3]
    #plt.imsave(fileImg, data_f, format = 'png')

    scipy.misc.toimage(data_f, cmin=0, cmax=255).save(fileImg)

# Wait for 5 seconds
    time.sleep(0.5)
    fileLab = MAIN_FOLDER + 'Labels/potsdam_' + attr + '.png'
    #plt.imsave(fileLab, label_p)
    scipy.misc.toimage(label_p, cmin=0, cmax=255).save(fileLab)
#
#fig = plt.figure()
#fig.add_subplot(121)
#plt.imshow(img)
