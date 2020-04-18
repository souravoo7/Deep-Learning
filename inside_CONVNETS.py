# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:58:14 2020

@author: sourav
"""

'''
Dissecting a CONVNET
'''
from keras.models import load_model
model = load_model('cats_and_dogs_base_v1.h5')
model.summary()

#get an image
img_path = 'G:/DATA SC/Deep_Learning/CONVNETS_EXT/base_dir/test/cats/cat.1700.jpg'

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)
#print the image
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()
'''
look at the model layers and channels
'''
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]] #the first 8 layers, get the outputs
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
print(first_layer_activation.shape)
#looking at thbe 3rd channel
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()
#looking at thbe 30th channel
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()

'''
Look at te activation channel outputs
'''
# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    
plt.show()




