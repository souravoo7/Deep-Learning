# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 02:46:14 2020

@author: sourav
"""

'''
Binary Classification of Moives based on the text content in reviews
Dataset: IMDB preprocessed data set containing the words mapped to integers
'''

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 1000)
'''
loading the top 10000 most frequently occuring data sets
would like to view the data shape and examples
'''

'''
Data Preprocessing: Set up data for ANN
'''

import numpy as np

def vectorize (sequences, dimension = 1000):
    results = np.zeros((len(sequences),dimension ))
    for i, sequence in enumerate (sequences):
        results [i, sequence] = 1.
    return results

x_train = vectorize(train_data)
x_test =  vectorize(test_data)


y_train = np.asarray((train_labels).astype('float32'))
y_test  = np.asarray((test_labels).astype('float32'))



'''
Build the ANN:
    1. The input data is vectors and the output is a scalar
    2. Going to build a dense (fully connected) network with relu activations
'''
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation = 'relu', input_shape=(1000,)))
model.add(layers.Dense(32, activation = 'relu')) 
# the input shape is adjusted by default in the next layers
model.add(layers.Dense(1, activation = 'sigmoid'))
#default pakacage functions
'''
model.compile(optimizer = 'rmsprop',
              loss ='binary_crossentropy',
              metrics = ['accuracy'])
'''
'''
The arguments are passed as string as these are available with the Keras Package
For customized functions need the optimizers class
'''

from keras import optimizers
from keras import losses
from keras import metrics

model.compile(optimizer = optimizers.RMSprop(lr = 0.001),
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])

'''
Create a validation data set
'''

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20,
                    batch_size = 512,
                    validation_data = (x_val, y_val))
'''
Post Fitting Analysis
'''
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt
'''
Plot the loss values
'''
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['binary_accuracy'])+1)
plt.plot (epochs,
          loss_values,
          'bo',
          label = 'Training Loss')

plt.plot (epochs,
          val_loss_values,
          'b',
          label = 'Validation Loss')

plt.title ('Training & Validation Loss')
plt.xlabel ('Epochs')
plt.ylabel ('Loss')
plt.legend()
plt.show()
plt.clf() #clear the figure

'''
Plot the accuracy values
'''

plt.clf() #clear the figure
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']

plt.plot (epochs,
          acc_values,
          'bo',
          label = 'Training Accuracy')

plt.plot (epochs,
          val_acc_values,
          'b',
          label = 'Validation Accuracy')

plt.title ('Training & Validation Accuracy')
plt.xlabel ('Epochs')
plt.ylabel ('Loss')
plt.legend()
plt.show()


'''
Model Evaluation on test data set and model predictions
'''
model.evaluate(x_test, y_test)
model.predict(x_test)


