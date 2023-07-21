# -*- coding: utf-8 -*-

from keras.datasets import mnist
from keras import models

import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy import misc



(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
image_index = 4444 # You may select anything up to 60,000
print(train_labels[image_index]) 
# =============================================================================
# create subplots 4 rows and 2 columns to display 4 different filters
# =============================================================================
f, axarr = plt.subplots(4,2)
plt.tight_layout()  #This just makes the titles for the subplots space properly

# =============================================================================
# Display the first two images before filters
# =============================================================================
axarr[0,0].imshow(train_images[image_index], cmap='Greys')
axarr[0,1].imshow(train_images[image_index], cmap='Greys')

# =============================================================================
# # vertical filter bright to dark
# =============================================================================

filter = np.array([[ 1, 0, -1],
                   [ 1, 0, -1],
                   [ 1, 0, -1]])

res = signal.convolve2d(train_images[image_index], filter)
axarr[1,0].set_title('Vertical B->D')
axarr[1,0].imshow(res, cmap='Greys')

# =============================================================================
# # vertical filter dark to bright
# =============================================================================
filter = np.array([[ -1, 0, 1],
                   [ -1, 0, 1],
                   [ -1, 0, 1]])

res = signal.convolve2d(train_images[image_index], filter)
axarr[2,0].set_title('Vertical D->B')
axarr[2,0].imshow(res, cmap='Greys')

# =============================================================================
# horizontal filter bright to dark
# =============================================================================
filter = np.array([[ 1, 1, 1],
                   [ 0, 0, 0],
                   [ -1, -1, -1]])

res = signal.convolve2d(train_images[image_index], filter)
axarr[1,1].set_title('Horizontal B->D')
axarr[1,1].imshow(res, cmap='Greys')

# =============================================================================
# horizontal filter dark to bright
# =============================================================================
filter = np.array([[ -1, -1, -1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]])

res = signal.convolve2d(train_images[image_index], filter)
axarr[2,1].set_title('Horizontal D->B')
axarr[2,1].imshow(res, cmap='Greys')

# =============================================================================
# # vertical filter sorbel
# =============================================================================

filter = np.array([[ 1, 0, -1],
                   [ 2, 0, -2],
                   [ 1, 0, -1]])

res = signal.convolve2d(train_images[image_index], filter)
axarr[3,0].set_title('Vertical Sorbel')
axarr[3,0].imshow(res, cmap='Greys')

# =============================================================================
# # vertical filter scharr
# =============================================================================
filter = np.array([[ 3, 0, -3],
                   [ 10, 0, -10],
                   [ 3, 0, -3]])

res = signal.convolve2d(train_images[image_index], filter)
axarr[3,1].set_title('Vertical Scharr')
axarr[3,1].imshow(res, cmap='Greys')