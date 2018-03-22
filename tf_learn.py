# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:12:40 2018

@author: arbaaz
"""

#from fcn_resnet import make_fcn_resnet, FCN_RESNET
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import (
    Input, Activation, Reshape, Conv2D, Lambda, Add)
import tensorflow as tf
from keras.utils.vis_utils import plot_model

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

input_tensor = Input(shape=(256, 256, 3))  # this assumes K.image_data_format() == 'channels_last'
nb_labels = 6
nb_rows = 256
nb_cols = 256

model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)


print(model.summary())
with open('report_keras.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
x32 = model.get_layer('activation_341').output
x16 = model.get_layer('activation_359').output
x8 = model.get_layer('activation_368').output

c32 = Conv2D(nb_labels, (1, 1), name='conv_labels_32')(x32)
c16 = Conv2D(nb_labels, (1, 1), name='conv_labels_16')(x16)
c8 = Conv2D(nb_labels, (1, 1), name='conv_labels_8')(x8)

def resize_bilinear(images):
    return tf.image.resize_bilinear(images, [nb_rows, nb_cols])

r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)

m = Add(name='merge_labels')([r32, r16, r8])

x = Reshape((nb_rows * nb_cols, nb_labels))(m)
x = Activation('softmax')(x)
x = Reshape((nb_rows, nb_cols, nb_labels))(x)

model = Model(inputs=input_tensor, outputs=x)

##########################
###Keras Documentations###
##########################

#model = make_fcn_resnet((256, 256, 3), use_pretraining=True, freeze_base=True, nb_labels=6)
x = base_model.output

x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new data for a few epochs
model.fit_generator(...)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)