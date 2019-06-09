from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.layers import LeakyReLU
from keras import optimizers

import keras
import pandas as pd 
import numpy as np
import cv2
import matplotlib.pyplot as plt

import os 
import time

df = pd.read_csv('velocity_labels.csv',\
names = ['image', 'velocity', 'steering_angle', 'outcome'],\
converters = {'image': lambda x: str(x), 'outcome': lambda x: '1' if x.strip() == 'good' else '0'})

print (df.tail()) # DEBUG


# Create training data labels and target

training_data = []
for img, label in zip(df['image'], df['steering_angle']):
    image = cv2.imread(f'output/{img}.png', 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # plt.imshow(image)
    # plt.show() 
    training_data.append([image, label])

X = [img for img, label in training_data]
Y = [label for img, label in training_data]

X = np.array(X)
Y = np.array(Y)


# Create Model 
model = Sequential()

# model.add(Flatten(input_shape = (940,940,3)))
model.add(Conv2D(5, (5, 5), strides=(2, 2), input_shape = (940,940,3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(5, (5, 5), strides=(2, 2)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(5, (5, 5), strides=(2, 2)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(5, (3, 3), strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(5, (3, 3), strides=1))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(1164)) 
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Activation('tanh'))

model.compile(loss='sparse_categorical_crossentropy',
metrics=['accuracy', 'mae', 'mse'], optimizer='sgd')

optimizers.SGD(lr=0.003, momentum=0.002)

history = model.fit(X, Y, batch_size= 30, epochs = 30, validation_split=0.3)

mse = history.history['mean_squared_error']
mae = history.history['mean_absolute_error']
acc = history.history['val_acc']

print(f'''
    {mse}
    {mae}
    {acc}''')

print(model.summary())



