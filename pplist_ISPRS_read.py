import tables
import numpy as np
from random import shuffle
from math import ceil
import matplotlib.pyplot as plt

hdf5_path = 'C:/Users/arbaaz/Desktop/Vision Project/isprs.h5'
batch_size = 10
subtract_mean = True
# open the hdf5 file
hdf5_file = tables.open_file(hdf5_path, mode='r')
# subtract the training mean
if subtract_mean:
    mm = hdf5_file.root.train_mean[0]
    mm = mm[np.newaxis, ...]
# Total number of samples
data_num = hdf5_file.root.DS.train_img.shape[0]

# create list of batches to shuffle the data
batches_list = list(range(int(ceil(float(data_num) / batch_size))))
shuffle(batches_list)
# loop over batches
for n, i in enumerate(batches_list):
    i_s = i * batch_size  # index of the first image in this batch
    i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
    # read batch images and remove training mean
    train_images = hdf5_file.root.DS.train_img[i_s:i_e]
    label_images = hdf5_file.root.DS.train_labels[i_s:i_e]
    if subtract_mean:
        train_images -= mm
    # read labels and convert to one hot encoding
#    labels = hdf5_file.root.train_labels[i_s:i_e]
#    labels_one_hot = np.zeros((batch_size, nb_class))
#    labels_one_hot[np.arange(batch_size), labels] = 1
#    print (n+1, '/', len(batches_list))
#    print (labels[0], labels_one_hot[0, :])
    plt.imshow(train_images[9])
    plt.show()
    plt.imshow(label_images[9])
    plt.show()
    
    if n == 5:  # break after 5 batches
        break
hdf5_file.close()