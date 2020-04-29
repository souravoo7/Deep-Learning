# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:06:46 2020

@author: sourav
"""
'''
Embedding from scratch
1. Train the embedding layer 
2. Feed the output to a dense classifier on top
'''
from keras.layers import Embedding

# The Embedding layer takes at least two arguments:
# the number of possible tokens, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64)

from keras.datasets import imdb
from keras import preprocessing

max_features = 10000 #number of words to be considered as features
maxlen = 20 #the 20 most common words

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
'''
from the 10000 most frequent, take first 20 words
'''
# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
'''
Define the model
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Embedding(10000, 8, input_length = maxlen)) #the embedding layer can only be the first layer in the Network
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'rmsprop',
              loss = 'binary_crossentropy',
              metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

print(history.history['val_acc'])


'''
Embedding using the Pre-Trained frameworks
1. Embdding layer is not learned here
2. The dense classifier is used on top
'''

'''
Read the data from the text files
'''
import os
import numpy as np
imdb_dir = 'G:/DATA SC/Deep_Learning/IMDB_RAW/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')


labels =[]
texts =[]

for label_type in ['neg', 'pos']:
    dir_name =os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f= open(os.path.join(dir_name, fname), encoding="utf-8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

'''
Tokenize the text data
'''
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 100  # We will cut reviews after 100 words
training_samples = 2000  # We will be training on 200 samples
validation_samples = 10000  # We will be validating on 10000 samples
max_words = 10000  # We will only consider the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index #list of word we have in the data
print('Found %s unique tokens.' % len(word_index))

#pad the data to fixed length of 100 
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
'''
Shuffle elements in the array
'''
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
'''
Create the Test and validation data sets
'''
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

'''
SET UP THE PRE_TRAINED EMBEDDINGS
'''

glove_dir = 'G:\DATA SC\Deep_Learning\GLOVE'

embeddings_index = {}

f = open('G:/DATA SC/Deep_Learning/GLOVE/glove.6B.100d.txt', encoding = "utf-8")

#get the word-vector map
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype= 'float32')
    embeddings_index[word] = coefs

f.close()
print('Found %s word vectors.' % len(embeddings_index))

'''
Prepare data to load on to the embedding layer
'''

embedding_dim = 100 # embedding dimension of 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)#get the vector from the pre-trained embedding vector for each of the corresponding words
    if i < max_words:#goes till 10000 words
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

'''
Define the NN model
'''

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words,
                    embedding_dim,
                    input_length = maxlen))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation= 'sigmoid'))

model.summary()

'''
LOAD GLOVE
'''
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False #these layers are locked
'''
Train the ANN
'''
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

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
RUN MODEL ON TEST DATA
'''
#read in the test data
test_dir = os.path.join(imdb_dir, 'test')
labels =[]
texts =[]
for label_type in ['neg', 'pos']:
    dir_name =os.path.join(test_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':#last four characters are  '.txt'
            f= open(os.path.join(dir_name, fname), encoding="utf-8")
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

#tokenize the test data
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, 
                       maxlen = maxlen)#till first 100 words
y_test = np.asarray(labels)

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)