import numpy as np
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
#from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import ( MaxPooling2D, Input, Activation, Reshape, Conv2D, 
                          Lambda, Add, merge, UpSampling2D, concatenate, Dropout)
# from keras.utils import np_utils
# from keras.optimizers import Adam
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from myDataFeederAerialFCN import RegDataGenerator
import os
import h5py

# 2. Constants
np.random.seed(123)  # for reproducibility

nb_labels = 6
nb_rows = 256
nb_cols = 256

inputs = Input((256, 256, 5))

conv1 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
print ("conv1 shape:",conv1.shape)
#conv1 = Conv2D(64, 5, activation = 'relu', dilation_rate=5, padding = 'same', kernel_initializer = 'he_normal')(conv1)
print ("conv1 shape:",conv1.shape)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
print ("pool1 shape:",pool1.shape)
drop1 = Dropout(0.5)(pool1)
#
conv2 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
print ("conv2 shape:",conv2.shape)
#conv2 = Conv2D(64, 5, activation = 'relu', dilation_rate=5, padding = 'same', kernel_initializer = 'he_normal')(conv2)
print ("conv2 shape:",conv2.shape)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
print ("pool2 shape:",pool2.shape)
drop2 = Dropout(0.5)(pool2)

conv3 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
print ("conv3 shape:",conv3.shape)
#conv3 = Conv2D(64, 5, activation = 'relu', dilation_rate=5, padding = 'same', kernel_initializer = 'he_normal')(conv3)
print ("conv3 shape:",conv3.shape)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
print ("pool3 shape:",pool3.shape)
##
conv4 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
#conv4 = Conv2D(128, 5, activation = 'relu', dilation_rate=5, padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
#pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#print ("pool4 shape:",pool4.shape)

conv5 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4)
#conv5 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
print ("conv5 shape:",conv5.shape)
drop5 = Dropout(0.5)(conv5)
#
up6 = Conv2D(64, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop3,up6], axis= 3)
print ("merge6 shape:",merge6.shape)

# merge([drop3,up6], mode = 'concat', concat_axis = 3)
conv6 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
#conv6 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
print ("conv6 shape:",conv6.shape)

up7 = Conv2D(64, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv2, up7], axis= 3)
print ("merge7 shape:",merge7.shape)

#merge7 = merge([conv2,up7], mode = 'concat', concat_axis = 3)
conv7 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
#conv7 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
print ("conv7 shape:",conv7.shape)

up8 = Conv2D(64, 5, activation = 'relu', padding = 'same', 
             kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv1, up8], axis= 3)
print ("merge8 shape:",merge8.shape)

#merge8 = merge([conv1,up8], mode = 'concat', concat_axis = 3)
conv8 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
#conv8 = Conv2D(32, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
print ("conv8 shape:",conv8.shape)

conv9 = Conv2D(16, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
print ("conv9 shape:",conv9.shape)

conv10 = Conv2D(1, 1, activation = 'relu')(conv9)

model = Model(inputs=inputs, outputs=conv10)

print(model.summary())

model.compile(loss='mse',
              optimizer='adam',
              metrics=[metrics.mean_squared_error, 
                       metrics.mean_absolute_error, 
                       metrics.mean_absolute_percentage_error, 
                       metrics.cosine_proximity]
              )

hdf5_path = 'C:/Users/arbaaz/Desktop/Deep Learning/Vision/Vision-Project/isprs_3.h5'
batch_size = 40

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
gp_val = hdf5_file['/DS1/val_img']
list_IDs_train = gp_train.shape[0]
list_IDs_val = gp_val.shape[0]
 
# 6. Preprocess class labels
params_train = {'dim_x': 256,
                'dim_y': 256,
                'dim_z': 5,
                'batch_size': batch_size,
                'shuffle': True,
                'hdf5_file': hdf5_file,
                'option': 'train'}

params_val = {'dim_x': 256,
              'dim_y': 256,
              'dim_z': 5,
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