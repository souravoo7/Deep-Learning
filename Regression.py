# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:40:39 2020

@author: sourav
"""
'''
Regression using ANN
'''
from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
'''
Preparing the data for regression: Feature Normalization (Z= X-mu/sigma)
'''
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)

train_data = train_data-mean
train_data = train_data/std

test_data = test_data-mean
test_data = test_data/std


from keras import models
from keras import layers

'''
The scalar regression model set up
'''
def build_model():
    model = models.Sequential()
    model.add(layers.Dense (64, activation = 'relu', input_shape =  (train_data.shape[1],)))
    model.add(layers.Dense (16, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', 
                  loss = 'mse', 
                  metrics= ['mae'])
    return model

'''
Setting up K-Fold Validation as the data set is small in size
'''
import numpy as np
k = 4
num_val_samples = len(train_data) // k

num_epochs = 80 #?? how to decide on this? higer MAE , look for higher epochs??
all_scores = []
all_MAE_hist = []

for i in range(k):
    print('processing fold#' , i)
    val_data    = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    
    partial_train_data      = np.concatenate([train_data[:i*num_val_samples], 
                                         train_data[(i+1)*num_val_samples:]], axis = 0)
    partial_train_targets   = np.concatenate([train_targets[:i*num_val_samples], 
                                         train_targets[(i+1)*num_val_samples:]], axis = 0)
    model = build_model()
    history = model.fit(partial_train_data,
              partial_train_targets,
              epochs = num_epochs,
              batch_size = 16,
              verbose = 0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
    all_scores.append(val_mae)
    MAE_hist = history.history['mae']
    all_MAE_hist.append(MAE_hist)
    
'''
Post - Fiiting Analysis
'''
print(np.mean(all_scores)) #the mean absolute validation error accross all the K-folds
avg_MAE_hist =   [np.mean([x[i] for x in all_MAE_hist])
                            for i in range(num_epochs)]  
#print(avg_MAE_hist)
 
import matplotlib.pyplot as plt
 
plt.plot(range(1, len(avg_MAE_hist)+1),avg_MAE_hist)
plt.xlabel('Epochs')
plt.ylabel('Validation_MAE')
plt.show() 

'''
Performance on Test data set
'''
results = model.evaluate(test_data, test_targets)   
print('Off-by $', abs(1000*results[1]))