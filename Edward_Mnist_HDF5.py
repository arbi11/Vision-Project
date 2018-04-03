import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from edward.models import Categorical, Normal
import edward as ed
#import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import edward as ed
from edward.models import Bernoulli, Empirical, Multinomial
from edward.util import Progbar
from keras.layers import Dense
from scipy.misc import imsave
import matplotlib.pyplot as plt
from edward.util import Progbar
import numpy as np
from skimage import img_as_float, img_as_int
#import h5py
from random import shuffle
import tables
from math import ceil
import sklearn.preprocessing

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

batch_size = 16
D = 256 * 256 * 3 # number of features 

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding= "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1,2,2,1], strides= [1,2,2,1], padding= "SAME")

x_image = tf.placeholder(tf.float32, shape = [batch_size, 256,256,3], name = "x_placeholder")
x = tf.reshape(x_image, [-1,196608])

with tf.name_scope("model"):
    
    W_conv1 = Normal(loc=tf.zeros([5,5,3,32]), scale=tf.ones([5,5,3,32]), name="W_conv1")
    b_conv1 = Normal(loc=tf.zeros([32]), scale=tf.ones([32]), name="b_conv1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1 )
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1.value()) + b_conv1.value() )    may be necessary
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = Normal(loc=tf.zeros([5,5,32,32]), scale=tf.ones([5,5,32,32]), name="W_conv2")
    b_conv2 = Normal(loc=tf.zeros([32]), scale=tf.ones([32]), name="b_conv2")
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    ####################################################
    ## Now the we have the min layer as a 64 x 64 x 32 = 131072 ##
    ####################################################
    
    W_conv3 = Normal(loc=tf.zeros([5,5,32,32]), scale=tf.ones([5,5,32,32]), name="W_conv3")
    b_conv3 = Normal(loc=tf.zeros([32]), scale=tf.ones([32]), name="b_conv3")
    deconv_shape_conv3 = tf.stack([batch_size, 64, 64, 32])
    
    W_conv4 = Normal(loc=tf.zeros([5,5,16,32]), scale=tf.ones([5,5,16,32]), name="W_conv3")
    b_conv4 = Normal(loc=tf.zeros([16]), scale=tf.ones([16]), name="b_conv3")
    deconv_shape_conv4 = tf.stack([batch_size, 128, 128, 16])
    
    W_conv5 = Normal(loc=tf.zeros([5,5,1,16]), scale=tf.ones([5,5,1,16]), name="W_conv3")
    b_conv5 = Normal(loc=tf.zeros([1]), scale=tf.ones([1]), name="b_conv3")
    deconv_shape_conv5 = tf.stack([batch_size, 256, 256, 1])
    
    #deconv_shape_conv5 = tf.pack([batch_size, 632, 632, 32])
    
    ## Now the conv2d_transpose part. Hopfuly just looking at the encoder part and 
    ## decoder part side by side will make it clear how it works.
    with tf.name_scope('deconv'):
        h_conv3 = tf.nn.relu(tf.nn.conv2d_transpose(h_pool2, W_conv3, output_shape = deconv_shape_conv3, 
                                  strides=[1,1,1,1], padding='SAME') + b_conv3)
        tf.summary.histogram('weights', W_conv3)
        tf.summary.histogram('activations', h_conv3)
    
    ##h_pool3 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv5, W_pool3, output_shape = deconv_shape_pool3, strides=[1,2,2,1], padding='SAME') + b_pool3)
    with tf.name_scope('deconv'):
        h_conv4 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv3, W_conv4, output_shape = deconv_shape_conv4, 
                                          strides=[1,2,2,1], padding='SAME') + b_conv4)
        tf.summary.histogram('weights', W_conv4)
        tf.summary.histogram('activations', h_conv4)
      
    ##h_pool4 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv6, W_pool4, output_shape = deconv_shape_pool4, strides=[1,2,2,1], padding='SAME') + b_pool4)
    with tf.name_scope('deconv'):
        h_conv5 = tf.nn.sigmoid(tf.nn.conv2d_transpose(h_conv4, W_conv5, output_shape = deconv_shape_conv5, 
                                          strides=[1,2,2,1], padding='SAME') + b_conv5)
        tf.summary.histogram('weights', W_conv5)
        tf.summary.histogram('activations', h_conv5)
    
# y = Categorical(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# y_conv = tf.zeros(h_conv5.shape, dtype=tf.float32)
#layer_shape = h_conv5.get_shape()
#num_features = layer_shape[1:4].num_elements()
#y_conv = tf.reshape(h_conv5, [-1])

y_conv = tf.reshape(h_conv5, [-1, 65536])
#log = [0.4, 0.8, 0.1, 0.5, 0.3, 0.2]

# y = Categorical(logits=neural_network(x))
#t = Categorical( logits= p_vec, sample_shape = I )

y = Categorical(logits=y_conv, sample_shape = batch_size)

with tf.name_scope("posterior_Normal"):
    qW_conv1 = Normal(loc = tf.Variable(tf.random_normal([5,5,3,32])), 
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([5,5,3,32]))))
    qb_conv1 = Normal(loc=tf.Variable(tf.random_normal([32])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([32]))))
    
    qW_conv2 = Normal(loc = tf.Variable(tf.random_normal([5,5,32,32])), 
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([5,5,32,32]))))
    qb_conv2 = Normal(loc=tf.Variable(tf.random_normal([32])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([32]))))
    
    qW_conv3 = Normal(loc = tf.Variable(tf.random_normal([5,5,32,32])), 
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([5,5,32,32]))))
    qb_conv3 = Normal(loc=tf.Variable(tf.random_normal([32])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([32]))))
    
    qW_conv4 = Normal(loc = tf.Variable(tf.random_normal([5,5,16,32])), 
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([5,5,16,32]))))
    qb_conv4 = Normal(loc=tf.Variable(tf.random_normal([16])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([16]))))
    
    qW_conv5 = Normal(loc = tf.Variable(tf.random_normal([5,5,1,16])), 
                    scale=tf.nn.softplus(tf.Variable(tf.random_normal([5,5,1,16]))))
    qb_conv5 = Normal(loc=tf.Variable(tf.random_normal([1])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

y_ = tf.placeholder(tf.int32, [batch_size, 256 *  256], name = "y_placeholder")

inference = ed.KLqp({W_conv1: qW_conv1, b_conv1: qb_conv1, W_conv2: qW_conv2, b_conv2: qb_conv2,
                      W_conv3: qW_conv3, b_conv3: qb_conv3, W_conv4: qW_conv4, b_conv4: qb_conv4, 
                      W_conv5: qW_conv5, b_conv5: qb_conv5 }, data={y: y_})

optimizer = tf.train.AdamOptimizer(0.01, epsilon=1.0)
inference.initialize(n_iter= 5000, n_print=100, optimizer=optimizer)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

hdf5_path = 'C:/Users/arbaaz/Desktop/Deep Learning/Vision/Vision-Project/isprs.h5'
hdf5_file = tables.open_file(hdf5_path, mode='r')
gp = '/DS'
folder = hdf5_file.get_node(gp)

label_binarizer = sklearn.preprocessing.LabelBinarizer()
label_binarizer.fit(range(7))

print('Starting Edward Training')

for _ in range(inference.n_iter):

    data_num_train = folder.train_img.shape[0]
    data_num_val = folder.val_img.shape[0]

    batches_list = list(range(int(ceil(float(data_num_train) / batch_size))))
    batches_list = batches_list[:-1]
    batches_val = list(range(int(ceil(float(data_num_val) / (batch_size)))))
    batches_val = batches_val[:-1]
    shuffle(batches_list)
    shuffle(batches_val)
    
    #for n, i in enumerate(batches_list):
    i=batches_list[1]
    
    # total_itr += 1
    i_s = i * batch_size  # index of the first image in this batch
    i_e = min([(i + 1) * batch_size, data_num_train])  # index of the last image in this batch
    # read batch images and remove training mean
    train_images = folder.train_img[i_s:i_e]
    label_images = folder.train_labels[i_s:i_e]
    Y_batch = np.zeros((label_images.shape[0], label_images.shape[1], label_images.shape[2]), dtype= np.uint8)
#    label = np.zeros((label_images.shape[0], label_images.shape[1], label_images.shape[2], 7), dtype= np.uint8)
    
    for j in range(batch_size):
        Y_batch[j,:,:] = convert_from_color(label_images[j,:,:,:])
#        for k in range(label_images.shape[1]):
#            label[j,k,:,:] = label_binarizer.transform(Y_batch[j,k,:])
#    #print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)
    Y_batch = np.reshape(Y_batch, [-1])
    info_dict =  inference.update(feed_dict= {x_image:train_images,  y_: Y_batch})
    inference.print_progress(info_dict)
    

##a = [1,0,3, 6]
#label_binarizer = sklearn.preprocessing.LabelBinarizer()
#label_binarizer.fit(range(7))
#b = label_binarizer.transform(a)
#print('{0}'.format(b))
#ed.set_seed(314159)