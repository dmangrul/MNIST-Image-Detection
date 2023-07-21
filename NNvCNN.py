#IMPORT THE LIBRARIES
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

#LOAD THE MNIST DATASET IN KERAS

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#DISPLAY ONE OF THE IMAGES USING MATPLOTLIB
image_index = 7777 # You may select anything up to 60,000
print(train_labels[image_index]) # The label is 8
plt.imshow(train_images[image_index], cmap='Greys')
one_image = train_images[image_index]

#VERIFY THE SIZE, SHAPE OF TRAINING AND TEST DATA
print ('Train Shape', train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)

#RESHAPE THE ARRAYS

test_images = test_images.reshape((10000,28,28,1))
train_images = train_images.reshape((60000,28,28,1))


#CONVERT GREYSCALE NUMBERS (0-255) TO A NUMBER BETWEEN 0-1
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
one_image = one_image.astype('float32')/255

#CONVERT THE Y VALUES (0-9) TO CATEGORICAL VALUES (LIKE ONE HOT ENCODING)
before_categ_test_labels = test_labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#CREATE THE NETWORK
"""
The following lines of code show you what a basic convnet looks 
like. It’s a stack of Conv2D and MaxPooling2D layers. 
"""

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))



"""
You can see that the output of every Conv2D and MaxPooling2D 
layer is a 3D tensor of shape (height, width, channels). 
The width and height dimensions tend to shrink as you go 
deeper in the network. The number of channels is controlled
by the first argument passed to the Conv2D layers (32 or 64).
"""

"""
The next step is to feed the last output tensor (of shape (3, 3, 64)) into a 
densely connected classifier network like those you’re already familiar with
: a stack of Dense layers. These classifiers process vectors, which are 1D,
 whereas the current output is a 3D tensor. First we have to flatten the 3D
 outputs to 1D, and then add a few Dense layers on top
 """
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.summary()  	

"""
As you can see, the (3, 3, 64) outputs are flattened into vectors of shape
 (576,) before going through two Dense layers.
 """

#COMPILE THE NETWORK
model.compile(optimizer = 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#TIME TO TRAIN THE NETWORK AND PRINT THE RESULTS
model.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
