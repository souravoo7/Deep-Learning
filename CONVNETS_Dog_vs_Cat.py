# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:18:18 2020

@author: sourav
"""

"""
Using Convnets to solve an image classification problem
Problem Statement: https://www.kaggle.com/c/dogs-vs-cats/overview

"""

'''
Download the data and split them to 3 parts : train, CV and test
Directory creation and data arrangement steps
'''
import os
import shutil

# The path to the directory where the original
# dataset was uncompressed
original_dataset_dir = 'G:/DATA SC/Deep_Learning/CONVNETS_EXT/train'

# The directory where we will
# store our smaller dataset
base_dir = 'G:/DATA SC/Deep_Learning/CONVNETS_EXT/base_dir'
os.mkdir(base_dir)

# Directories for our training,
# validation and test splits
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Directory with our validation cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# Directory with our validation dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

'''
Copy the images to the appropriate directories
'''

# Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


"""
Count the data transferred to new locations: Sanity check
We have a 1000, 500, 500 split for TRAIN, CV and TEST for Dogs and Cats respectively
Total 2000*2 = 4000 images
"""
cats_train_size = len(os.listdir(train_cats_dir))

'''
Creating the base network model
'''

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,
                        (3,3),
                        activation = 'relu',
                        input_shape = (150,150,3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid')) #binary_classification

model.summary()

'''
Compile the model
'''

from keras import optimizers
from keras import losses
#from keras import metrics

model.compile(optimizer = optimizers.RMSprop(lr = 1e-4),
              loss = losses.binary_crossentropy,
              metrics = ['acc'])

'''
Data Preprocessing:
Currently, our data sits on a drive as JPEG files, so the steps for getting it into our network are roughly:

    Read the picture files.
    Decode the JPEG content to RBG grids of pixels.
    Convert these into floating point tensors.
    Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know, neural networks prefer to deal with small input values).

'''

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255, scaling the pixel from (0,255) range to [0,1] range
#define the pyhton generators from keras image processing toolkit
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, #target directory
        target_size = (150,150),#resize images to 150 x 150 as demanded by the ANN 
        batch_size =20, # data will be in batches of size 20
        class_mode = 'binary')#class mode is binary as the data has 2 labels

validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size = (150,150),
        batch_size =20,
        class_mode = 'binary')

#information about the data generated
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 30,
        validation_data = validation_generator,
        validation_steps =50)

model.save('cats_and_dogs_base_v1.h5') 

'''
Post-Fitting Analysis:Get the training and validation accuracy plots
Check for over-fitting
'''
import matplotlib.pyplot as plt

history.history.keys() #check the model metric keys loss and accuracy
#get the accuracy and loss metrics
acc =  history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')

plt.title ('Training & Validation Accuracy')
plt.xlabel ('Epochs')
plt.ylabel ('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title ('Training & Validation Loss')
plt.xlabel ('Epochs')
plt.ylabel ('Loss')
plt.legend()

plt.show()

'''
Data Augumentation to resolve the Over-Fitting Problems
Data Augmentation involves creating new data samples from the existing samples using transformations

datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
The new images are related to the original images that are transformed, hence it may not help greatly with the over-fitting

'''

'''
Re-designing the ANN with data augmentation and drop-out (50%)
'''

model = models.Sequential()
model.add(layers.Conv2D(32,
                        (3,3),
                        activation = 'relu',
                        input_shape = (150,150,3)))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(128, (3,3), activation = 'relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid')) #binary_classification

model.summary()

'''
Compile the model
'''
model.compile(optimizer = optimizers.RMSprop(lr = 1e-4),
              loss = losses.binary_crossentropy,
              metrics = ['acc'])

'''
using data augumentation: the image transformer
'''
train_datagen = ImageDataGenerator(
        rescale             =   1./255,
        rotation_range      =    40,
        width_shift_range   =   0.2,
        height_shift_range  =   0.2,
        shear_range         =   0.2,
        zoom_range          =   0.2,
        horizontal_flip     =   True)

#test/validation generator created
#we are not using the image transformer here as we do not do data augmentaion on thevalidation and test data sets
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150, 150),
        batch_size = 32, # ?can train using smaller number of batches per epoch
        class_mode = 'binary'
        )
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size = (150, 150),
        batch_size = 32, # ?can train using smaller number of batches per epoch
        class_mode = 'binary')
'''
Model Fitting
'''
history = model.fit_generator(
        train_generator,
        steps_per_epoch =100,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = 50)

model.save('cats_and_dogs_base_v2.h5')

acc =  history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')

plt.title ('Training & Validation Accuracy')
plt.xlabel ('Epochs')
plt.ylabel ('Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')

plt.title ('Training & Validation Loss')
plt.xlabel ('Epochs')
plt.ylabel ('Loss')
plt.legend()

plt.show()    