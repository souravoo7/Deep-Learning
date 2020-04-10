# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 01:48:49 2020

@author: sourav
"""

from keras.datasets import mnist


"""
data stored in 4 numpy arrays of train and test data
"""
(train_images, train_labels), (test_images, test_labels) =mnist.load_data()
#looking at the data
train_images.shape
len(train_labels)
train_labels

#build the network 

from keras import models
from keras import layers

network = models.Sequential()
#add 2 dense layers (fully connected)
network.add(layers.Dense(512, 
                         activation='relu', 
                         input_shape=(28*28,)))
network.add(layers.Dense(10, 
                         activation='softmax'))
"""
1. Add a loss function to measure the performance on the training data
2. An optimizer to update the network after each iteration/training
3. Metrics for testing: model accuracy
"""
network.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

'''
Images are to be reshaped for training the network
28*28 to match the input shape of the network
'''
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float32')/255

'''
Change the labels to categorical data

'''
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels) 


'''
Fit the network
'''

network.fit(train_images,
            train_labels,
            epochs = 5,
            batch_size =128)

'''
Evaluate the network
'''
test_loss, test_acc = network.evaluate(test_images, test_labels)

print ('Accuracy:', test_acc)