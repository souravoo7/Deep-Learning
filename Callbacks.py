# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:29:36 2020

@author: sourav
"""
import tensorflow as tf
"""
CALLBACKS in KERAS
1. Model Check Point and Early Stoppage
"""

'''
Applying on IMDB dataset
'''

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers

max_features = 2000
maxlen = 500
batch_size = 32

print ('loading data.....')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences.......')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model = tf.keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, 
                           input_length = maxlen,
                           name = 'embed'))

model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.MaxPooling1D(5))

model.add(layers.Conv1D(32, 7, activation = 'relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

"""
Define the Callbacks to be used: Unsing in-built callbacks
"""
callbacks_list_A = [
    tf.keras.callbacks.EarlyStopping( #early stoppage 
        monitor = 'acc', #monitor the metric
        patience=1),#monitor for the number of iterations
    tf.keras.callbacks.ModelCheckpoint( #save the model given the conditions match
        filepath = 'model_checkpoint.h5',
        monitor = 'val_loss', #conditioned on the monitoring of validation loss
        save_best_only = True), #save the model with the best validation loss
    tf.keras.callbacks.ReduceLROnPlateau( #change the learinng rate
        monitor = 'val_loss', # via monitoring the validation loss
                    factor = 0.1,# change by factor of 0.1, reduction
                    patience = 4)# change after 4 iterations/epocs of no improvement
    ]

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop (lr = 1e-4), 
              loss = 'binary_crossentropy', 
              metrics = ['acc'])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    callbacks = callbacks_list_A, # include the required callbacks here
                    validation_split=0.3) # for call backs require the monitoring of validation metrics, validation data is important


'''
Post-Fitting Analysis: Training and Validation Curves
'''

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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
Tensor Board Visulaization
'''
callbacks_list_B = [
    tf.keras.callbacks.TensorBoard(
        log_dir = 'C:/Users/sourav/log_test',
        histogram_freq = 1,
        embeddings_freq = 1)
    ]

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    callbacks = callbacks_list_B, # include the required callbacks here
                    validation_split=0.3) # for call backs require the monitoring of validation metrics, validation data is important
