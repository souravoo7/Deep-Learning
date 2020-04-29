# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 02:54:55 2020

@author: sourav
"""

'''
Advanced RNN for Time-Series Forecasting
-------------------------------------------------------------------------------

About the data set:
A weather timeseries dataset recorded at the Weather Station at the Max-Planck-Institute for Biogeochemistry in Jena, Germany: http://www.bgc-jena.mpg.de/wetter/.

In this dataset, fourteen different quantities (such air temperature, atmospheric pressure, humidity, wind direction, etc.) are recorded every ten minutes, over several years. The original data goes back to 2003, but we limit ourselves to data from 2009-2016.We will use it to build a model that takes as input some data from the recent past (a few days worth of data points) and predicts the air temperature 24 hours in the future.
'''

"""
Data Set-up and exploration
"""

import os

data_dir = 'G:/DATA SC/Deep_Learning/JENA_CLIMATE'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f=open(fname)#open file
data =f.read()#read file
f.close()#close file

lines = data.split('\n') # line break to identify the lines
header = lines[0].split(',')# get the data header
lines = lines[1:] # store the rest of the data exclusing the header
print (len(header))
print(len(lines))

#conver the data into a numpy array
import numpy as np
float_data = np.zeros((len(lines), len(header) -1))

for i , line in enumerate(lines):
    values= [float(x) for x in line.split(',')[1:]] # drop the fist column
    float_data [i,: ] = values


#look at the data points
from matplotlib import pyplot as plt
temp = float_data[:, 1] # the temperature column
plt.plot(range(len(temp)), temp)
plt.show()
#look at a shorter range
plt.plot(range(1000), temp[:1000])
plt.show()

'''
Data Preprocessing for ANN input
1. Normalization of data using the mean and variance approach
2. Preparation of prediction samples
'''
mean = float_data[:200000].mean(axis = 0) # will be using the first 200k data points for training 
float_data = float_data-mean
std = float_data[:200000].std(axis = 0)
float_data = float_data/std

'''

    data: The original array of floating point data, which we just normalized in the code snippet above.

    lookback: How many timesteps back should our input data go.

    delay: How many timesteps in the future should our target be.

    min_index and max_index: Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another one for testing.

    shuffle: Whether to shuffle our samples or draw them in chronological order.

    batch_size: The number of samples per batch.

    step: The period, in timesteps, at which we sample data. We will set it 6 in order to draw one data point every hour.

'''

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
#------------------------------------------------------------------------------
lookback = 1440 # 10 days
step = 6 #1 hour
delay = 144 # 1 day
batch_size = 500

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size = batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size = batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size

'''
Using the today - tomorrow thumb rule
'''
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return (np.mean(batch_maes))

t_mae = evaluate_naive_method()
temp_mae =std[1]*t_mae 
'''
Using a basic dense network
'''
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
'''
Post Fitting Analysis
'''
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


'''
Using a Recurrent Neural Network
1. We will use the GRU (gated recurrent unit) network
2.It is compuationally cheaper that LSTM but not that efficient (trade-off)
3.Will use the drop out rate of 20% in both the input layers as well as in the the recurrent layers
4.Networks with drop out takes longer time to converge, hence the epocs should be more.
'''
model = Sequential()
model.add(layers.GRU(32,
                     dropout = 0.2, #input drop out
                     recurrent_dropout = 0.2,#recurrent layer drop out rate
                     input_shape = (None, float_data.shape[-1])))

model.add(layers.Dense(1, activation ='softmax'))

model.compile(optimizer = RMSprop(lr = 0.0005),
              loss = 'mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=200,
                              epochs=30,
                              validation_data=val_gen,
                              validation_steps=val_steps)

'''
Post Fitting Analysis
'''

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


'''
Using a multi-layered RNN
'''

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.3,
                     return_sequences=True,#this parameter is necessary
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='softmax',
                     dropout=0.2, 
                     recurrent_dropout=0.2))
model.add(layers.Dense(1, activation = 'softmax'))

model.compile(optimizer=RMSprop(lr = 0.0005), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

'''
Post Fitting Analysis
'''

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss- Multi Layered RNN')
plt.legend()

plt.show()

'''
Using a Bi-directional RNN
'''

model = Sequential()
model.add(layers.Bidirectional(
    layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

'''
Post Fitting Analysis
'''

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss - Bi-Directional RNN')
plt.legend()

plt.show()


'''
USING CNN AND RNN for processing
'''
#resuing the generator for higher resolution
lookback = 720 # 5 days
step = 3 #1 hour
delay = 144 # 1 day
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size = batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size = batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size

'''
MODEL DEFINITION
'''
model = Sequential()
model.add(layers.Conv1D(32, 
                        5, 
                        activation = 'relu', 
                        input_shape = (None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 
                        5, 
                        activation = 'relu'))
          
model.add(layers.GRU(32,
                     dropout = 0.2, 
                     recurrent_dropout = 0.2))

model.add(layers.Dense(1))

model.summary()

model.compile(optimizer = RMSprop(lr = 1e-4),
              loss = 'mae',
              metrics = ['acc'])

history = model.fit_generator(train_gen,
                              steps_per_epoch=200,
                              epochs=15,
                              validation_data=val_gen,
                              validation_steps=val_steps)

'''
Post Fitting Analysis
'''

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation Loss - CNN+RNN')
plt.legend()

plt.show()