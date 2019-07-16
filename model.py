from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.layers import LeakyReLU, Concatenate, Input
from keras import optimizers
from keras.utils import plot_model

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import keras
import pandas as pd
import numpy as np
import cv2

import os
import time
import pickle


# ########################################################
# init DF for training data
df = pd.read_csv('velocity_labels.csv',\
names = ['image', 'velocity', 'steering_angle', 'outcome'],\
converters = {'image': lambda x: str(x), 'outcome': lambda x: '1' if x.strip() == 'good' else '0',\
              'steering_angle': lambda x: round(float(x)/70, 8),\
              'velocity': lambda x: round(float(x.strip()), 8)})

df['normal_velocity'] = round(((df['velocity'] - min(df['velocity']))/ (max(df['velocity']) - min(df['velocity']))), 8)
df['normal_steering_angle'] = round(((df['steering_angle'] - min(df['steering_angle']))/ (max(df['steering_angle']) - min(df['steering_angle']))), 8)

# #########################################################
# Saving normalised training data to csv
df.to_csv('normal_dataset.csv', encoding='utf-8', index=False)
print (df.tail()) # DEBUG

# ########################################################
# init DF for test data
test_df = pd.read_csv('test_labels.csv',\
names = ['image', 'velocity', 'steering_angle', 'outcome'],\
converters = {'image': lambda x: str(x).strip(), 'outcome': lambda x: '1' if x.strip() == 'good' else '0',\
                'steering_angle': lambda x: round(float(x)/70, 8),\
                'velocity': lambda x: round(float(x.strip()), 8)})

test_df['normal_velocity'] = round(((test_df['velocity'] - min(test_df['velocity']))/ (max(test_df['velocity']) - min(test_df['velocity']))), 8)
test_df['normal_steering_angle'] = round(((test_df['steering_angle'] - min(test_df['steering_angle']))/ (max(test_df['steering_angle']) - min(test_df['steering_angle']))), 8)

# ########################################################
# PARAMETERS

BS = 600 # batch size
batch_per_epoch = np.ceil(len(df['image']) / BS).astype(int)
epochs = 100
alpha = 0.01
lr = 0.00001
mom = 0.00002

# ########################################################
# Create training data labels and target

def training_gen():

    while True:
        training_data = []
        for img, velocity, angle, outcome, normal_v, normal_angle in df.values:
            try:
                if len(training_data) != BS:

                    image = cv2.imread(f"small_training_images/{img}.png", 1)
                    #image = cv2.resize(image, (400,400))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # plt.imshow(image)
                    # plt.show()
                    training_data.append([image, angle, normal_v])

                else:
                    X = [image for image, angle, normal_v in training_data]
                    other_inp = [image for image, angle, normal_v in training_data]
                    Y = [image for image, angle, normal_v in training_data]

                    training_data = []

                    X = np.array(image).reshape(-1, 400,400,3)
                    Y = np.array([angle])
                    other_inp = np.array([velocity])

                    yield [X, other_inp], [Y]


            except cv2.error:
                pass

# ########################################################
# Create test data labels and target
test_data = []
for img, angle, outcome, velocity in zip(test_df['image'], test_df['steering_angle'], test_df['outcome'], test_df['normal_velocity']):
    try:
        test_image = cv2.imread(f"small_test_images/{img}.png", 1)
        #test_image = cv2.resize(test_image, (400,400))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # plt.imshow(image)
        # plt.show()
        if outcome == '1':
            test_data.append([test_image, angle, velocity])

    except cv2.error:
        pass

test_X = [img for img, label, velocity in test_data]
test_Y = [label for img, label, velocity in test_data]
test_other_inp = [velocity for img, label, velocity in test_data]

test_X = np.array(test_X).reshape(-1, 400,400,3)
test_Y = np.array(test_Y)
test_other_inp = np.array(test_other_inp)
print(test_other_inp.shape)

# ########################################################
# Create Model
def create_model():
    # model = Sequential()
    input1 = Input(shape=(400,400,3))

    # model.add(Flatten(input_shape = (940,940,3)))
    conv1 = Conv2D(20, (5, 5), strides=(2, 2))(input1)
    #conv1 = model.add(Conv2D(20, (5, 5), strides=(2, 2), input_shape = (940,940,3)))
    activ1 =LeakyReLU(alpha=alpha)(conv1)
    pool1 =MaxPooling2D(pool_size=(2, 2))(activ1)

    conv2 = Conv2D(16, (5, 5), strides=(2, 2))(pool1)
    activ2 =LeakyReLU(alpha=alpha)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(activ2)

    # conv3 = Conv2D(12, (5, 5), strides=(2, 2))(pool2)
    # activ3 = LeakyReLU(alpha=alpha)(conv3)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(activ3)

    # conv4 = Conv2D(8, (3, 3), strides=1)(pool3)
    # activ4 = LeakyReLU(alpha=alpha)(conv4)
    # pool4 = MaxPooling2D(pool_size=(1, 1))(activ4)

    # conv5 = Conv2D(4, (3, 3), strides=1)(pool4)
    # activ5 = LeakyReLU(alpha=alpha)(conv5)
    # pool5 = MaxPooling2D(pool_size=(1, 1))(activ5)

    flat1 = Flatten()(pool2)

    input2 = Input(shape = (1, ))
    concat1 = Concatenate(axis=1)([flat1, input2])

    dense1 = Dense(1164, activation='tanh')(concat1)
    dense2 = Dense(500, activation='tanh')(dense1)
    dense3 = Dense(200, activation='tanh')(dense2)
    dense4 = Dense(1, activation='tanh')(dense3)
    #activ6 = Activation('softmax')(dense4)

    model = Model(inputs = [input1, input2], outputs = dense4)

    model.compile(loss='kullback_leibler_divergence',
    metrics=['accuracy', 'mae', 'mse'], optimizer='sgd')

    optimizers.SGD(lr=lr, momentum=mom)

    #history = model.fit([X, other_inp], Y, batch_size= 40, epochs = 50, validation_data=([test_X, test_other_inp], test_Y))
    history = model.fit_generator(training_gen(),\
            steps_per_epoch=batch_per_epoch, epochs=epochs, \
            validation_data=([test_X, test_other_inp], test_Y))


    return (history, model)
# ############################################################
# Call Function
history, model = create_model()

mse = history.history['mean_squared_error']
mae = history.history['mean_absolute_error']
acc = history.history['acc']
loss = history.history['loss']

print(f'''
    mse: {min(mse)}
    mae: {min(mae)}
    acc: {max(acc)}''')

model.summary()

plot_model(model, show_shapes=True, show_layer_names=False)

# Dump the model to the pickle file

with open('model.pickle', 'wb') as output:
    pickle.dump(model, output)


loss_graph = plt.plot(loss)
plt.show()
