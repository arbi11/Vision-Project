import numpy as np
from skimage import img_as_float

class RegDataGenerator(object):
  'Generates data for Keras'
  def __init__(self, hdf5_file, option, subtract_mean = True, dim_x = 256, dim_y = 256, dim_z = 3, batch_size = 4, shuffle = True):
      'Initialization'
      self.dim_x = dim_x
      self.dim_y = dim_y
      self.dim_z = dim_z
      self.batch_size = batch_size
      self.shuffle = shuffle
      #self.hdf5_path = hdf5_path
      self.hdf5_file = hdf5_file
      self.subtract_mean = subtract_mean
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
              y = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
              # Generate data
              # gp = '/DS'
              # folder = self.hdf5_file.get_node(gp)

########################
######### H5Py #########
########################

              gp_train_img = self.hdf5_file['/DS/train_img']
              gp_val_img = self.hdf5_file['/DS/val_img']
              gp_train_lab = self.hdf5_file['/DS/train_labels'] 
              gp_val_lab = self.hdf5_file['/DS/val_labels']
              
#              if self.subtract_mean:
#                  mean = self.hdf5_file['/train_mean'][0]
#                  #mean = mean[np.newaxis, ...]

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
#                      if self.subtract_mean:
#                          a -= mean
                      a = img_as_float(a) 
                      #a.astype('float32')
                      #a /= 255.0
                      X[i, :, :, :] = a
                      # y[i, :] = folder.train_perfs[ID, 7:10]
                      a = gp_train_lab[ID]
                      a = img_as_float(a)
#                      a = a.astype('float32')
#                      a /= 255.0
                      y[i, :, :, :] = a
              elif self.option == 'val':
                  for i, ID in enumerate(list_IDs_temp):
                      # a = folder.val_labels[ID]
                      a = gp_val_img[ID]
#                      if self.subtract_mean:
#                          a -= mean
                      a = img_as_float(a)
#                      a = a.astype('float32')
#                      a /= 255.0
                      X[i, :, :, :] = a
                      # y[i, :] = folder.val_perfs[ID, 7:10]
                      a = gp_val_lab[ID]
                      a = img_as_float(a)
#                      a = a.astype('float32')
#                      a /= 255.0
                      y[i, :, :, :] = a
              
              yield X, y