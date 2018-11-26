# -*- coding: utf-8 -*-
"""Alireza Behtash

"""All packages needed"""

import shutil
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import time
import os
import sys
import zipfile
import skimage.transform
import imageio
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16


# upload the simpsons dataset (the dataset you just downloaded to your local computer). 

mainpath = '/Users/...'
dest_filename = os.path.join(mainpath, 'the-simpsons-dataset.zip')


def extract(filename, force=False):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .zip
    if os.path.isdir(root) and not force:
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        zip = zipfile.ZipFile(filename)
        sys.stdout.flush()
        zip.extractall(mainpath)
        zip.close()
        print("Extraction done!")
    return root

#main folder containing the entire dataset (after unzipping the file uploaded)
data_folder = extract(dest_filename)


"""#Prepare the dataset"""

#Change image shape to a uniform one across all dataset
img_w = 160   #image width
img_h = 160    #image height
new_shape = (img_h, img_w, 3)

def read_img_train(location):
    path = os.path.abspath(data_folder + '/train')  #absolute path of the folder
    path = re.sub('[a-zA-Z\s._]+$', '', path)   #removing the absolute path to only focus on the file name
    #assigning arrays to train dataset 
    x_train = [] 
    y_train = []  
    #label number will be stored in an array
    label_name = []
    #list of all directories in which images can be found for training purposes
    dirs_train = os.listdir(data_folder + '/train/')  
    #label starts at 0 and ends at 3 since we only have a total of 4 classes
    label = 0  
    #loop over all directories
    for i in dirs_train:  
        if i != '.DS_Store':   #an extra irrelevant class will otherwise appear that is hidden. 
          count = 0  #counting data in each folder
          label_name.append(i)  # save folder name in var label_name
          for n in glob.glob(data_folder + '/train/' + i + '/*.jpg'):  # loop all data
              read_image = imageio.imread(n)
              im = skimage.transform.resize(read_image, new_shape, preserve_range = False )
              im = np.array(im)  #store im as numpy array
              R = im[:, :, 0]
              G = im[:, :, 1]
              B = im[:, :, 2]
              x_train.append([R, G, B])  #save RGB in x_train
              y_train.append([label])  #save in y_train
              count = count + 1  #inner loop continues to the next count
          label = label + 1  #outer loop continues to the next label
    return np.array(x_train), np.array(y_train)

def read_img_test(location):
    path = os.path.abspath(data_folder + '/test')  #absolute path of the folder
    path = re.sub('[a-zA-Z\s._]+$', '', path)   #removing the absolute path to only focus on the file name
    #assigning arrays to test dataset 
    x_test = [] 
    y_test = []  
    #label number will be stored in an array
    label_name = []
    #list of all directories in which images can be found for testing purposes
    dirs_test = os.listdir(data_folder + '/test/')  
    #label starts at 0 and ends at 3 since we only have a total of 4 classes
    label = 0  
    #loop over all directories
    for i in dirs_test:  
        if i != '.DS_Store':   #an extra irrelevant class will otherwise appear that is hidden. 
          count = 0  #counting data in each folder
          label_name.append(i)  # save folder name in var label_name
          for n in glob.glob(data_folder + '/test/' + i + '/*.jpg'):  # loop all data
              read_image = imageio.imread(n)
              im = skimage.transform.resize(read_image, new_shape, preserve_range = False )
              im = np.array(im)  #store im as numpy array
              R = im[:, :, 0]
              G = im[:, :, 1]
              B = im[:, :, 2]
              x_test.append([R, G, B])  #save RGB in x_test
              y_test.append([label])  #save in y_test
              count = count + 1  #inner loop continues to the next count
          label = label + 1  #outer loop continues to the next label
    return np.array(x_test), np.array(y_test)

def read_img_val(location):
    path = os.path.abspath(data_folder + '/validation')  #absolute path of the folder
    path = re.sub('[a-zA-Z\s._]+$', '', path)   #removing the absolute path to only focus on the file name
    #assigning arrays to validation dataset 
    x_val = [] 
    y_val = []  
    #label number will be stored in an array
    label_name = []
    #list of all directories in which images can be found for validating purposes
    dirs_val = os.listdir(data_folder + '/validation/')  
    #label starts at 0 and ends at 3 since we only have a total of 4 classes
    label = 0  
    #loop over all directories
    for i in dirs_val:  
        if i != '.DS_Store':   #an extra irrelevant class will otherwise appear that is hidden. 
          count = 0  #counting data in each folder
          label_name.append(i)  # save folder name in var label_name
          for n in glob.glob(data_folder + '/validation/' + i + '/*.jpg'):  # loop all data
              read_image = imageio.imread(n)
              im = skimage.transform.resize(read_image, new_shape, preserve_range = False )
              im = np.array(im)  #store im as numpy array
              R = im[:, :, 0]
              G = im[:, :, 1]
              B = im[:, :, 2]
              x_val.append([R, G, B])  #save RGB in x_val
              y_val.append([label])  #save in y_val
              count = count + 1  #inner loop continues to the next count
          label = label + 1  #outer loop continues to the next label
    return np.array(x_val), np.array(y_val)


num_class = 4 #num of classes/labels

#Data normalization
x_train, y_train = read_img_train('train/') #call read_img
x_test, y_test = read_img_test('test/')
x_val, y_val = read_img_val('validation/')

x_train = x_train.reshape(x_train.shape[0], img_w, img_h, 3) #reshape x_train into: (num of data, 416,288,3)
x_test = x_test.reshape(x_test.shape[0], img_w, img_h, 3) #reshape x_test into (num of data, 416, 288,3)
x_val = x_val.reshape(x_val.shape[0], img_w, img_h, 3) #reshape x_test into (num of data, 416, 288,3)

x_train = x_train.astype('float64') #type set to float64
x_test = x_test.astype('float64')
x_val = x_val.astype('float64')

y_train_cat = keras.utils.to_categorical(y_train, num_class) #change y_train into categorical like [0,1,0...,0]
y_test_cat = keras.utils.to_categorical(y_test, num_class) # change y_test into categorical
y_val_cat = keras.utils.to_categorical(y_val, num_class) # change y_val into categorical

"""#Generate augmented data using ImageDataGenerator"""

batch_size = 40
epochs = 30

#Data generator for augmentation
datagen = train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
        rescale= 1. / 225)    #Important: we don't need to augment the validation AND test sets
train_generator = datagen.flow_from_directory(data_folder + '/train',  batch_size=batch_size, target_size=(img_w, img_h),
                                              class_mode='categorical', shuffle = True)
val_generator = val_datagen.flow_from_directory(data_folder + '/validation', batch_size=batch_size, target_size=(img_w, img_h),
                                              class_mode='categorical', shuffle = True)
test_generator = val_datagen.flow_from_directory(data_folder + '/test', batch_size=batch_size, target_size=(img_w, img_h),
                                              class_mode='categorical', shuffle = True)

"""#Define the model"""

#Model definition
model = Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=new_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(num_class, activation='softmax'))
model.summary()

"""#Train the model"""

#Fit the model to data
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])
history = model.fit_generator(
      train_generator,
      epochs=epochs,
      validation_data=val_generator,
      steps_per_epoch=250, validation_steps=250)

"""#Testing the trained model on the test data"""

loss, acc = model.evaluate_generator(test_generator, verbose=0, steps = 500)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

"""## The base model performance"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

nb_epochs = range(len(acc))


plt.plot(nb_epochs, acc, 'bo', label='Training acc')
plt.plot(nb_epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(nb_epochs, loss, 'bo', label='Training loss')
plt.plot(nb_epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""#Using a pretrained base model"""

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(160, 160, 3))
conv_base.summary()
print('Conv_Base Summary')

conv_base.trainable = False

model = Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_class, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(
      train_generator,
      epochs=epochs,
      validation_data=val_generator,
      steps_per_epoch=250, validation_steps=250)

loss, acc = model.evaluate_generator(test_generator, verbose=0, steps = 500)
print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

nb_epochs  = range(len(acc))


plt.plot(nb_epochs, acc, 'bo', label='Training acc')
plt.plot(nb_epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(nb_epochs, loss, 'bo', label='Training loss')
plt.plot(nb_epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
