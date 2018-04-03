import numpy as np
from skimage import img_as_float
from keras.utils import to_categorical

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

def one_hot(data, n_classes):
    return (to_categorical(data, num_classes=n_classes))

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

class RegDataGenerator(object):
  'Generates data for Keras'
  def __init__(self, hdf5_file, option, dim_x = 256, dim_y = 256, dim_z = 3, batch_size = 16, n_classes = 7, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle
      #self.hdf5_path = hdf5_path
      self.hdf5_file = hdf5_file
      self.option = option
                
  def generate(self, list_IDs):
      'Generating batches of samples'
      
      while 1:
          indexes = np.arange(list_IDs)
          np.random.shuffle(indexes)
          imax = int(len(indexes)/self.batch_size)
          
          for i in range(imax):
              list_IDs_temp = indexes[i*self.batch_size : (i+1)*self.batch_size]
              
              X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
              y = np.empty((self.batch_size, self.dim_x * self.dim_y))
########################
######### H5Py #########
########################

              gp_train_img = self.hdf5_file['/DS/train_img']
              gp_val_img = self.hdf5_file['/DS/val_img']
              gp_train_lab = self.hdf5_file['/DS/train_labels'] 
              gp_val_lab = self.hdf5_file['/DS/val_labels'] 

########################
##### PyTables #########
########################              
            
#              gp_train_img = self.hdf5_file.root.DS.train_img
#              gp_val_img = self.hdf5_file.root.DS.val_img
#              gp_train_lab = self.hdf5_file.root.DS.train_labels 
#              gp_val_lab = self.hdf5_file.root.DS.val_labels

############### Common for both ###########################
              if self.option == 'train':
                  for i, ID in enumerate(list_IDs_temp):
                      a = gp_train_img[ID]
                      a = img_as_float(a)
                      X[i, :, :, :] = a
                      
                      a = gp_train_lab[ID]
                      a = convert_from_color(a)
                      a = a.reshape([256*256])/6
#                      a = a.astype('int32')
#                      a = one_hot(a, n_classes = self.n_classes)
                      y[i, :] = a
                      
              elif self.option == 'val':
                  for i, ID in enumerate(list_IDs_temp):
                      # a = folder.val_labels[ID]
                      a = gp_val_img[ID]
                      a = a.astype('float32')
                      a /= 255.0
                      X[i, :, :, :] = a
                      a = gp_val_lab[ID]
                      a = convert_from_color(a)
                      a = a.reshape([256*256])/6
#                      a = a.astype('int32')
#                      a = one_hot(a, n_classes = self.n_classes)
                      y[i, :] = a
              
              yield X, y