# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 23:39:55 2020

@author: sourav
"""
'''
One-hot-Encoding technique at at a word level
'''
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index)+1
            
max_length =10 #max number of words to be considered in each sequence

#define the numeric tensor to be used for string the code

results =np.zeros(shape = (len(samples),#this gives the number of strings/sentences present
                           max_length, #number of words in each sentence to be considered
                           max(token_index.values())+1))#number of unique words in the samples            

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j,index] = 1.
        
'''
One-hot-Encoding technique at at a character level
'''
import string

characters = string.printable #set of all printable ASCII characters

token_index = dict(zip(range(1, len(characters)+1), 
                       characters))
max_length =50 #max number of characters to be considered in each sequence

results = np.zeros((len(samples),
                    max_length,
                    max(token_index.keys())+1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i,j,index] = 1.


'''
Using keras in built functions for ONE_HOT_ENCODING
'''
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words = 1000)#take into account most common 1000 words

tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

#key step to convert text to matrix
one_hot_results = tokenizer.texts_to_matrix(samples, mode ='binary')

word_index = tokenizer.word_index
print(len(word_index))





