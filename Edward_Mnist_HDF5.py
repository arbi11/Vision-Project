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
from edward.models import Bernoulli, Normal, Categorical,Empirical, Multinomial
from edward.util import Progbar
from keras.layers import Dense
from scipy.misc import imsave
import matplotlib.pyplot as plt
from edward.util import Progbar
import numpy as np
from skimage import img_as_float


hdf5_path = 'C:/Users/arbaaz/Desktop/Deep Learning/dataset_acer.h5'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#ed.set_seed(314159)
N = 100   # number of images in a minibatch.
D = 784   # number of features.
K = 10    # number of classes.
batch_size = 16

x = tf.placeholder(tf.float32, [None, D])
# Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
# Categorical likelihood for classication.
y = Categorical(tf.matmul(x,w)+b)

# Contruct the q(w) and q(b). in this case we assume Normal distributions.
qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

# We use a placeholder for the labels in anticipation of the traning data.
y_ph = tf.placeholder(tf.int32, [N])
# Define the VI inference technique, ie. minimise the KL divergence between q and p.
inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})
# Initialse the infernce variables
inference.initialize(n_iter=5000, n_print=100, scale={y: float(mnist.train.num_examples) / N})

# We will use an interactive session.
sess = tf.InteractiveSession()
# Initialise all the vairables in the session.
tf.global_variables_initializer().run()
# Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
for _ in range(inference.n_iter):
    X_batch, Y_batch = mnist.train.next_batch(N)
    # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
    Y_batch = np.argmax(Y_batch,axis=1)
    info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
    inference.print_progress(info_dict)

N = 500  # number of data points
D = 256 * 256 * 3 # number of features 

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1,1,1,1], padding= "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize= [1,2,2,1], strides= [1,2,2,1], padding= "SAME")


x = tf.placeholder(tf.float32, shape = [N, 196608], name = "x_placeholder")
#y_ = tf.placeholder("float", shape = [None, 10])
y_ = tf.placeholder(tf.int32, [N], name = "y_placeholder")

x_image = tf.reshape(x, [-1,256,256,3])


#with tf.name_scope("model"):
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
deconv_shape_conv3 = tf.stack([batch_size, 64, 64, 32)

W_conv4 = Normal(loc=tf.zeros([5,5,32,16]), scale=tf.ones([5,5,32,16]), name="W_conv3")
b_conv4 = Normal(loc=tf.zeros([16]), scale=tf.ones([16]), name="b_conv3")
deconv_shape_conv4 = tf.stack([batch_size, 128, 128, 16)

W_conv5 = Normal(loc=tf.zeros([5,5,16,3]), scale=tf.ones([5,5,16, 3]), name="W_conv3")
b_conv5 = Normal(loc=tf.zeros([3]), scale=tf.ones([3]), name="b_conv3")
deconv_shape_conv5 = tf.stack([batch_size, 256, 256, 3)

#deconv_shape_conv5 = tf.pack([batch_size, 632, 632, 32])

## Now the conv2d_transpose part. Hopfuly just looking
## at the encoder part and decoder part side by side
## will make it clear how it works.
with tf.name_scope('deconv'):
    h_conv3 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv2, W_conv3, output_shape = deconv_shape_conv3, 
                              strides=[1,1,1,1], padding='VALID') + b_conv3)
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
    h_conv5 = tf.nn.relu(tf.nn.conv2d_transpose(h_conv4, W_conv5, output_shape = deconv_shape_conv5, 
                                      strides=[1,2,2,1], padding='SAME') + b_conv5)
    tf.summary.histogram('weights', W_conv5)
    tf.summary.histogram('activations', h_conv5)

# y = Categorical(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y = Multinomial(tf.matmul(h_conv5, W_conv5) + b_conv5)

# number of samples 
# we set it to 20 because of the memory constrain in the GPU. My GPU can take upto about 200 samples at once. 

T = 20
# INFERENCE
with tf.name_scope("posterior"):
    qW_conv1 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,5,5,3,32])))
    qb_conv1 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,32])))

    qW_conv2 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,5,5,32,32])))
    qb_conv2 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,32])))

    qW_conv3 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,5,5,32,32])))
    qb_conv3 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,32])))

    qW_conv4 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,5,5,32,16])))
    qb_conv4 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,16])))
    
    qW_conv4 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,5,5,16,3])))
    qb_conv4 = Empirical(params = tf.Variable(1/1000 *tf.random_normal([T,3])))

#X_batch , Y_batch = mnist.train.next_batch(N)
#Y_batch = np.argmax(Y_batch, axis = 1)
#dropout = 1.0
#inference = ed.KLqp({w: qw, b: qb}, data={y:y_ph})
## Initialse the infernce variables
#inference.initialize(n_iter=5000, n_print=100, scale={y: float(mnist.train.num_examples) / N})


inference = ed.SGHMC({W_conv1: qW_conv1, b_conv1: qb_conv1, W_conv2: qW_conv2, b_conv2: qb_conv2,
                      W_conv3: qW_conv3, b_conv3: qb_conv3, W_conv4: qW_conv4, b_conv4: qb_conv4, 
                      W_conv5: qW_conv5, b_conv5: qb_conv5 }, data={y: y_})
    
inference.initialize()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(inference.n_iter):

    
    info_dict_hmc =  inference.update(feed_dict= {x:X_batch,  y_: Y_batch, keep_prob : dropout})
    inference.print_progress(info_dict_hmc)