import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train = x_train / 255
x_test = x_test / 255


model = Sequential()

# first layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32,32,3)))

# second layer (pooling)
model.add(MaxPooling2D(pool_size=(2,2)))

# next convolution layer
model.add(Conv2D(32, (5,5), activation='relu'))

# next pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# flattening layer
model.add(Flatten())

# 1500 neuron layer
model.add(Dense(1500, activation='relu'))

# dropout layer
model.add(Dropout(0.5))

# 1000 neuron layer
model.add(Dense(1000, activation='relu'))

# dropout layer
model.add(Dropout(0.5))

# 500 neuron layer
model.add(Dense(500, activation='relu'))

# 10 neuron layer
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Model training
hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.2)

# evaluating model with test data set
print(model.evaluate(x_test, y_test_one_hot)[1])

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.show()

# Testing the file
