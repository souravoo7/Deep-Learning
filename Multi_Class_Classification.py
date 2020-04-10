# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 02:52:44 2020

@author: sourav
"""
'''
Multi-Class Classification 
Dataset: Reuters preprocessed data set containing the words mapped to integers
'''
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)

'''
Prepare the data for feeding into neural network
'''

import numpy as np

def vectorize (sequences, dimension = 10000):
    results = np.zeros((len(sequences),dimension ))
    for i, sequence in enumerate (sequences):
        results [i, sequence] = 1.
    return results

x_train = vectorize(train_data)
x_test =  vectorize(test_data)

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels) 

"""
Define the model
"""

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(128, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

'''
Validation data set-up
'''

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = train_labels[:1000]
partial_y_train = train_labels[1000:]

'''
Model Fitting
'''
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs =20, #important parameter to fix overfitting
                    batch_size =512,
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
epochs = range(1, len(history_dict['accuracy'])+1)
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

'''
Plot the accuracy values
'''

plt.clf() #clear the figure
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

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
model.evaluate(x_test, test_labels)
#model.predict(x_test)
