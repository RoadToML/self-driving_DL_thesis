from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.layers import LeakyReLU, Concatenate, Input
from keras import optimizers
from keras.utils import plot_model

import keras
import pandas as pd
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os
import time
import pickle

df = pd.read_csv('velocity_labels.csv',\
names = ['image', 'velocity', 'steering_angle', 'outcome'],\
converters = {'image': lambda x: str(x), 'outcome': lambda x: '1' if x.strip() == 'good' else '0',\
                'steering_angle': lambda x: round(float(x)/70, 8)})

df['normal_velocity'] = round(((df['velocity'] - min(df['velocity']))/ (max(df['velocity']) - min(df['velocity']))), 8)

df.to_csv('normal_dataset.csv', encoding='utf-8', index=False)
print (df.tail()) # DEBUG


# Create training data labels and target

training_data = []
for img, angle, outcome, velocity in zip(df['image'], df['steering_angle'], df['outcome'], df['velocity']):
    try:
        image = cv2.imread(f"output//{img}.png", 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # plt.imshow(image)
        # plt.show()
        if outcome == '1':
            training_data.append([image, angle, velocity])
    except cv2.error:
        pass

X = [img for img, label, velocity in training_data]
Y = [label for img, label, velocity in training_data]
other_inp = [velocity for img, label, velocity in training_data]

X = np.array(X).reshape(-1, 940,940,3)
Y = np.array(Y)
other_inp = np.array(other_inp)
print(other_inp.shape)


# Create Model
def create_model(X, Y, other_inp):
    # model = Sequential()

    input1 = Input(shape=(940,940,3))

    # model.add(Flatten(input_shape = (940,940,3)))
    conv1 = Conv2D(20, (5, 5), strides=(2, 2))(input1)
    #conv1 = model.add(Conv2D(20, (5, 5), strides=(2, 2), input_shape = (940,940,3)))
    activ1 =LeakyReLU()(conv1)
    pool1 =MaxPooling2D(pool_size=(2, 2))(activ1)

    conv2 = Conv2D(16, (5, 5), strides=(2, 2))(pool1)
    activ2 =LeakyReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(activ2)

    conv3 = Conv2D(12, (5, 5), strides=(2, 2))(pool2)
    activ3 = LeakyReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(activ3)

    conv4 = Conv2D(8, (3, 3), strides=1)(pool3)
    activ4 = LeakyReLU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(activ4)
    
    conv5 = Conv2D(4, (3, 3), strides=1)(pool4)
    activ5 = LeakyReLU()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(activ5)
    
    flat1 = Flatten()(pool5)

    input2 = Input(shape = (1, ))
    concat1 = Concatenate(axis=1)([flat1, input2])
    dense1 = Dense(1164, activation='tanh')(concat1)
    dense2 = Dense(100, activation='tanh')(dense1)
    dense3 = Dense(50, activation='tanh')(dense2)
    dense4 = Dense(10, activation='tanh')(dense3)
    #activ6 = Activation('softmax')(dense4)

    model = Model(inputs = [input1, input2], outputs = dense4)

    model.compile(loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'mae', 'mse'], optimizer='sgd')

    optimizers.SGD(lr=0.003, momentum=0.002)

    history = model.fit([X, other_inp], Y, batch_size= 20, epochs = 50, validation_split=0.2)

    return (history, model)

history, model = create_model(X, Y, other_inp)

mse = history.history['mean_squared_error']
mae = history.history['mean_absolute_error']
acc = history.history['acc']

print(f'''
    mse: {max(mse)}
    mae: {max(mae)}
    acc: {max(acc)}''')

model.summary()

plot_model(model, show_shapes=True, show_layer_names=False)

# Dump the model to the pickle file

with open('model.pickle', 'wb') as output:
    pickle.dump(model, output)
