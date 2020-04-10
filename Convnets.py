# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 00:02:22 2020

@author: sourav
"""

"""
Introducing CONVNETS: Convoluted Neural Networks
"""

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()
#convert the output tensor to a 1D tensor 
model.add(layers.Flatten()) #Flatten() to feed it into a dense layerd network
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax')) #10 digits to predict

model.summary()

"""
RUN  THIS ON THE MNIST DATA SET
"""

from keras.datasets import mnist

"""
data stored in 4 numpy arrays of train and test data
"""
(train_images, train_labels), (test_images, test_labels) =mnist.load_data()

"""
1. Add a loss function to measure the performance on the training data
2. An optimizer to update the network after each iteration/training
3. Metrics for testing: model accuracy
"""
model.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

'''
Images are to be reshaped for training the network
28*28 to match the input shape of the network
'''
train_images = train_images.reshape((60000,28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape(10000, 28, 28, 1)
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

model.fit(train_images,
            train_labels,
            epochs = 5,
            batch_size =64)

'''
Evaluate the network
'''
test_loss, test_acc = model.evaluate(test_images, test_labels)

print ('Accuracy:', test_acc)


