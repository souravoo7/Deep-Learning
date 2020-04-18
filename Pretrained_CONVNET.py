# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 02:42:59 2020

@author: sourav
"""


'''
Using a pretrained convoluted ANN: 
    A pre-trained network is simply a saved network previously trained on a large dataset, typically on a large-scale image classification task. If this original dataset is large enough and general enough, then the spatial feature hierarchy learned by the pre-trained network can effectively act as a generic model of our visual world, and hence its features can prove useful for many different computer vision problems, even though these new problems might involve completely different classes from those of the original task.
    
    VGG16 architecture, developed by Karen Simonyan and Andrew Zisserman in 2014, a simple and widely used convnet architecture for ImageNet
'''
#get the pretrained CONVNET from keras library
from keras.applications import VGG16

conv_base = VGG16( weights = 'imagenet',#weights from the imagenet learning
                   include_top = False, #do not bring in the dense layer
                   input_shape = (150, 150, 3))# input vector shape, canbe left blank to auto-adjust
conv_base.summary()
'''
Data Pre-processing
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = 'G:/DATA SC/Deep_Learning/CONVNETS_EXT/base_dir'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 20

'''
Feature Extraction: The convnet layers are not added into the model as it is computationally expensive (GPU required)
'''
def extract_features(directory, sample_count):
    '''
    Directory on which the feature extraction will occur
    Sample_count: number of samples
    '''
    features = np.zeros(shape = (sample_count, 4,4,512))#features from the convnet comes out with he sample_count x 4 x 4 x 512 dimensions, us the model summary to get the info
    labels = np.zeros(shape = (sample_count)) #one label for each sample
    generator = datagen.flow_from_directory(
            directory,
            target_size = (150,150),
            batch_size = batch_size,
            class_mode = 'binary') #image data preprocessing using the python keras image generators
    i=0 #initialize variables for generator break conditions
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)#conv_base from pretrained network using the new input data
        #each batch has the same features, so they are added batch wise
        features[i*batch_size:(i+1)*batch_size] = features_batch #add the features from the CONV_BASE
        labels[i*batch_size:(i+1)*batch_size] = labels_batch #labels remain same
        i=i+1
        if i*batch_size>=sample_count:
            break
    return  features, labels
#get the features
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

#reshape the data in flat arrays
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

'''
Write  the dense model layers above it
'''
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5)) #dropout of 50% added
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

#model fitting

history = model.fit(train_features,#these are outputs from the pretrained convnet 
                    train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

model.save('cats_and_dogs_base_v3.h5')

'''
Post-Fitting Analysis
'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

'''
Fine Tuning
'''
'''
#unlocking the fine tuning layers
conv_base.trainable = True
set_trainable =False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
'''

'''
need to re-run the entire model with fine tune layers unlocked: Need GPU power
#refit the model now with the unlocked parts
history = model.fit(train_features,#these are outputs from the pretrained convnet 
                    train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))
'''

'''
Evaluate model on test data
'''
test_loss, test_acc = model.evaluate(test_features, test_labels)
print ('Accuracy%' , 100*test_acc)

