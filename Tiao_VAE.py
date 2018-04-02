import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from keras import backend as K

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply, Conv2D, MaxPooling2D, Flatten
from keras.models import Model, Sequential
#from keras.datasets import mnist
from myDataFeederVAE_Tiao import RegDataGenerator
#import tables
import h5py
#original_dim = 784
intermediate_dim = 256
latent_dim = 2
batch_size = 100
epochs = 50
epsilon_std = 1.0

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

hdf5_path = 'D:/Documents/Codes/Vision/Vision-Project/isprs_3.h5'

img_size = 256

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

# Parameters
params_train = {'dim_x': 256,
                'dim_y': 256,
                'dim_z': 5,
                'batch_size': 32,
                'shuffle': True,
                'hdf5_file': hdf5_file,
                'option': 'train'}

params_val = {'dim_x': 256,
              'dim_y': 256,
              'dim_z': 5,
              'batch_size': 32,
              'shuffle': True,
              'hdf5_file': hdf5_file,
              'option': 'val'}

# Generators
training_generator = RegDataGenerator(**params_train).generate(list_IDs_train)
validation_generator = RegDataGenerator(**params_train).generate(list_IDs_train)

decoder = Sequential([
    Dense(intermediate_dim, input_dim=latent_dim, activation='relu'),
    Dense(256*256, activation='sigmoid')
])

#x = Input(shape=(original_dim,))
inputs = Input((256, 256, 5))

conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
print ("conv1 shape:",conv1.shape)
conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
print ("conv1 shape:",conv1.shape)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
print ("pool1 shape:",pool1.shape)

conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
print ("conv2 shape:",conv2.shape)
conv2 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
print ("conv2 shape:",conv2.shape)
pool2 = MaxPooling2D(pool_size=(4, 4))(conv2)
print ("pool2 shape:",pool2.shape)

conv3 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
print ("conv3 shape:",conv3.shape)
conv3 = Conv2D(4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
print ("conv3 shape:",conv3.shape)
#drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
print ("pool3 shape:",pool3.shape)

flat = Flatten()(pool3)
h = Dense(intermediate_dim, activation='relu')(flat)

z_mu = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
z_sigma = Lambda(lambda t: K.exp(.5*t))(z_log_var)

eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                   shape=(K.shape(inputs)[0], latent_dim)))
z_eps = Multiply()([z_sigma, eps])
z = Add()([z_mu, z_eps])

y_pred = decoder(z)

vae = Model(inputs=[inputs, eps], outputs=y_pred)
vae.compile(optimizer='rmsprop', loss=nll)

# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(-1, original_dim) / 255.
#x_test = x_test.reshape(-1, original_dim) / 255.
#
#vae.fit_generator(generator = training_generator,
#                  steps_per_epoch = list_IDs_train//batch_size,
#                  )

outputFolder = './output-VAE'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, \
                             save_best_only=False, save_weights_only=False, \
                             mode='auto', period=1)

early_stop = EarlyStopping(monitor='val_loss', patience=2)

callbacks_list = [checkpoint, early_stop]

hist = vae.fit_generator(generator = training_generator,
                         steps_per_epoch = list_IDs_train//batch_size,
                         validation_data = validation_generator,
                         validation_steps = list_IDs_train//batch_size,
                         epochs= 15,
                         verbose=1,
                         callbacks= callbacks_list
                         )
print(hist.history)
print('\n history showed')

vae.save('C:/Users/arbaaz/Desktop/Deep Learning/Keras/Keras/model_vae.h5')
print('\nModel saved')

#vae.fit(x_train,
#        x_train,
#        shuffle=True,
#        epochs=epochs,
#        batch_size=batch_size,
#        validation_data=(x_test, x_test))

encoder = Model(inputs, z_mu)

# display a 2D plot of the digit classes in the latent space
z_test = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
            alpha=.4, s=3**2, cmap='viridis')
plt.colorbar()
plt.show()

# display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28

# linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian to produce values
# of the latent variables z, since the prior of the latent space
# is Gaussian
u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
                               np.linspace(0.05, 0.95, n)))
z_grid = norm.ppf(u_grid)
x_decoded = decoder.predict(z_grid.reshape(n*n, 2))
x_decoded = x_decoded.reshape(n, n, digit_size, digit_size)

plt.figure(figsize=(10, 10))
plt.imshow(np.block(list(map(list, x_decoded))), cmap='gray')
plt.show()