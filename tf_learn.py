# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:12:40 2018

@author: arbaaz
"""

#from fcn_resnet import make_fcn_resnet, FCN_RESNET
import numpy as np
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
#from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import (
    Input, Activation, Reshape, Conv2D, Lambda, Add, merge, UpSampling2D, concatenate, Dropout)
# from keras.layers import MaxPooling2D, Conv2D, merge, UpSampling2D, concatenate
# from keras.utils import np_utils
# from keras.optimizers import Adam
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from myDataFeederSegnet import RegDataGenerator
import os
import h5py

# import tensorflow as tf
# from keras.utils.vis_utils import plot_model

#from keras.applications.inception_v3 import InceptionV3
#from keras.preprocessing import image
#from keras.layers import Dense, GlobalAveragePooling2D
#from keras import backend as K

# 2. Constants
np.random.seed(123)  # for reproducibility

# create the base pre-trained model
#base_model = InceptionV3(weights='imagenet', include_top=False)

input_tensor = Input(shape=(256, 256, 3))  # this assumes K.image_data_format() == 'channels_last'
nb_labels = 6
nb_rows = 256
nb_cols = 256

#model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

#model_vgg = VGG16()
base_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

#print(base_model.summary())
#with open('report_vgg.txt','w') as fh:
#    # Pass the file handle in as a lambda function to make it callable
#    base_model.summary(print_fn=lambda x: fh.write(x + '\n'))
  
B1_C2 = base_model.get_layer('block1_conv2').output    
B2_C2 = base_model.get_layer('block2_conv2').output
B3_C2 = base_model.get_layer('block3_conv2').output
B4_C4 = base_model.get_layer('block4_conv4').output
B5_C4 = base_model.get_layer('block5_conv4').output
conv_enc = base_model.get_layer('block5_pool').output

up6 = Conv2D(512, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_enc))

merge6 = concatenate([B5_C4,up6], axis= 3)
#print ("merge6 shape:",merge6.shape)

# merge([drop3,up6], mode = 'concat', concat_axis = 3)
conv6 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Dropout(0.5)(conv6)
conv6 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
drop6 = Dropout(0.5)(conv6)
print ("conv6 shape:",drop6.shape)

up7 = Conv2D(512, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop6))
merge7 = concatenate([B4_C4, up7], axis= 3)
print ("merge7 shape:",merge7.shape)

#merge7 = merge([conv2,up7], mode = 'concat', concat_axis = 3)
conv7 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Dropout(0.5)(conv7)
conv7 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
drop7 = Dropout(0.5)(conv7)
print ("conv7 shape:",drop7.shape)

up8 = Conv2D(256, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop7))
merge8 = concatenate([B3_C2, up8], axis= 3)
print ("merge8 shape:",merge8.shape)

#merge8 = merge([conv1,up8], mode = 'concat', concat_axis = 3)
conv8 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Dropout(0.5)(conv8)
conv8 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
drop8 = Dropout(0.5)(conv8)
print ("conv8 shape:",drop8.shape)

up9 = Conv2D(128, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop8))
merge9 = concatenate([B2_C2, up9], axis= 3)
print ("merge8 shape:",merge8.shape)

conv9 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
drop9 = Dropout(0.5)(conv9)
print ("conv9 shape:",drop9.shape)

up10 = Conv2D(64, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop9))
print ("conv10 shape:",up10.shape)
merge10 = concatenate([B1_C2, up10], axis= 3)
print ("merge10 shape:",merge10.shape)

conv10 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
conv10 = Dropout(0.5)(conv10)
conv10 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv10)
drop10 = Dropout(0.5)(conv10)
print ("conv10 shape:",drop10.shape)

conv11 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop10)
drop11 = Dropout(0.5)(conv11)
conv11 = Conv2D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop11)

conv12 = Conv2D(3, 3, activation = 'relu', padding = 'same')(conv11)

model = Model(inputs=input_tensor, outputs=conv12)
#
#print(model.summary())

for layer in base_model.layers:
    layer.trainable = False
   
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.compile(loss='mse',
              optimizer='adam',
              metrics=[metrics.mean_squared_error, 
                       metrics.mean_absolute_error, 
                       metrics.mean_absolute_percentage_error, 
                       metrics.cosine_proximity]
              )


# hdf5_path = 'D:/Documents/Codes/CNN/CNN/dataset_acer2.h5'
# hdf5_path = 'D:/Documents/Codes/Keras/Keras/Uniform resolution data/dataset_acer3.h5'
hdf5_path = 'C:/Users/arbaaz/Desktop/Vision Project/isprs.h5'
# hdf5_path = '/home/arbaaz/Codes/Data/Unique/dataset_acer3.h5'
# nClasses = 3
batch_size = 40
#epochs = 3

# 5. Preprocess input data
########################
##### PyTables #########
########################

#hdf5_file = tables.open_file(hdf5_path, mode='r')
#gp = '/DS'
#folder = hdf5_file.get_node(gp)
#list_IDs_train = folder.train_img.shape[0]
#list_IDs_val = folder.val_img.shape[0]

########################
######### H5Py #########
########################

hdf5_file = h5py.File(hdf5_path, 'r')
gp_train = hdf5_file['/DS/train_img']
gp_val = hdf5_file['/DS/val_img']
# folder = hdf5_file.get_node(gp)
list_IDs_train = gp_train.shape[0]
list_IDs_val = gp_val.shape[0]
 
# 6. Preprocess class labels
#Y_train = np_utils.to_categorical(y_train, 10)
#Y_test = np_utils.to_categorical(y_test, 10)
 

# Parameters
params_train = {'dim_x': 256,
                'dim_y': 256,
                'dim_z': 3,
                'batch_size': batch_size,
                'shuffle': True,
                'hdf5_file': hdf5_file,
                'option': 'train'}

params_val = {'dim_x': 256,
              'dim_y': 256,
              'dim_z': 3,
              'batch_size': batch_size,
              'shuffle': True,
              'hdf5_file': hdf5_file,
              'option': 'val'}


# 5. Generators
training_generator = RegDataGenerator(**params_train).generate(list_IDs_train)
validation_generator = RegDataGenerator(**params_val).generate(list_IDs_val)


# 8. Fit model on training data
outputFolder = './output-Segnet-1'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                             save_best_only=False, save_weights_only=False, \
                             mode='auto', period=1)

early_stop = EarlyStopping(monitor='val_loss', patience=2)

callbacks_list = [checkpoint, early_stop]

hist = model.fit_generator(generator = training_generator,
                    steps_per_epoch = list_IDs_train//batch_size,
                    validation_data = validation_generator,
                    validation_steps = list_IDs_val//batch_size,
                    epochs= 15,
                    verbose=1,
                    callbacks= callbacks_list
                    )
print(hist.history)
print('\n history showed')

model.save('C:/Users/arbaaz/Desktop/Vision Project/model_Segnet_1.h5')
print('\nModel saved')

# serialize model to JSON
model_json = model.to_json()
with open("model_Segnet_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_Segnet_1.h5")
print("Saved model to disk")

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model_Segnet_1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model_Segnet_1.h5")
print("Saved model to disk")
 
# del model

#x32 = model.get_layer('activation_341').output
#x16 = model.get_layer('activation_359').output
#x8 = model.get_layer('activation_368').output
#
#c32 = Conv2D(nb_labels, (1, 1), name='conv_labels_32')(x32)
#c16 = Conv2D(nb_labels, (1, 1), name='conv_labels_16')(x16)
#c8 = Conv2D(nb_labels, (1, 1), name='conv_labels_8')(x8)
#
#def resize_bilinear(images):
#    return tf.image.resize_bilinear(images, [nb_rows, nb_cols])
#
#r32 = Lambda(resize_bilinear, name='resize_labels_32')(c32)
#r16 = Lambda(resize_bilinear, name='resize_labels_16')(c16)
#r8 = Lambda(resize_bilinear, name='resize_labels_8')(c8)
#
#m = Add(name='merge_labels')([r32, r16, r8])
#
#x = Reshape((nb_rows * nb_cols, nb_labels))(m)
#x = Activation('softmax')(x)
#x = Reshape((nb_rows, nb_cols, nb_labels))(x)
#
#model = Model(inputs=input_tensor, outputs=x)
#
###########################
####Keras Documentations###
###########################
#
##model = make_fcn_resnet((256, 256, 3), use_pretraining=True, freeze_base=True, nb_labels=6)
#x = base_model.output
#
#x = GlobalAveragePooling2D()(x)
## let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
## and a logistic layer -- let's say we have 200 classes
#predictions = Dense(200, activation='softmax')(x)
#
## this is the model we will train
#model = Model(inputs=base_model.input, outputs=predictions)
#
## first: train only the top layers (which were randomly initialized)
## i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
#    layer.trainable = False
#
## compile the model (should be done *after* setting layers to non-trainable)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#
## train the model on the new data for a few epochs
#model.fit_generator(...)
#
## at this point, the top layers are well trained and we can start fine-tuning
## convolutional layers from inception V3. We will freeze the bottom N layers
## and train the remaining top layers.
#
## let's visualize layer names and layer indices to see how many layers
## we should freeze:
#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
#
## we chose to train the top 2 inception blocks, i.e. we will freeze
## the first 249 layers and unfreeze the rest:
#for layer in model.layers[:249]:
#   layer.trainable = False
#for layer in model.layers[249:]:
#   layer.trainable = True
#
## we need to recompile the model for these modifications to take effect
## we use SGD with a low learning rate
#from keras.optimizers import SGD
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#
## we train our model again (this time fine-tuning the top 2 inception blocks
## alongside the top Dense layers
#model.fit_generator(...)