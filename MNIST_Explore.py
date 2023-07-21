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
test_images = test_images.reshape((10000, 28*28))
train_images = train_images.reshape((60000, 28*28))

#CONVERT GREYSCALE NUMBERS (0-255) TO A NUMBER BETWEEN 0-1
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255
one_image = one_image.astype('float32')/255

#CONVERT THE Y VALUES (0-9) TO CATEGORICAL VALUES (LIKE ONE HOT ENCODING)
before_categ_test_labels = test_labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#CREATE THE NETWORK
network = models.Sequential()

#ADD THE LAYERS - 1
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))

#ADD THE SECOND AND LAST LAYER
network.add(layers.Dense(10, activation='softmax'))

#COMPILE THE NETWORK
network.compile(optimizer = 'rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#TIME TO TRAIN THE NETWORK AND PRINT THE RESULTS
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print (network.summary())

predict_test = network.predict(test_images)
predictedProbs = np.amax(predict_test, axis=1)
predictedt = np.argmax(predict_test, axis=1)
errors = np.array(np.nonzero(before_categ_test_labels - predictedt))
wrongProbs = predictedProbs[errors]

fig1, ax1 = plt.subplots(2,3)

for i, ax1 in enumerate(ax1.flatten()):
    image_index = errors[0][np.argmax(wrongProbs)]
    print(np.amax(wrongProbs))
    print(str(predictedt[image_index]), "  " , str(before_categ_test_labels[image_index]))
    pixels = test_images[image_index].reshape((28, 28))
    ax1.imshow(pixels, cmap='Greys')
    wrongProbs = np.delete(wrongProbs, np.argmax(wrongProbs))

rights = np.array(np.where(predictedt == before_categ_test_labels))
rightProbs = predictedProbs[rights]
highprobs = np.array(np.where(rightProbs > .999)[1])

fig1, ax1 = plt.subplots(2,3)

for i, ax1 in enumerate(ax1.flatten()):
    image_index = rights[0][np.argmax(highprobs)]
    print(str(predictedt[image_index]), "  " , str(before_categ_test_labels[image_index]))
    pixels = test_images[image_index].reshape((28, 28))
    ax1.imshow(pixels, cmap='Greys')
    highprobs = np.delete(highprobs, np.argmax(highprobs))