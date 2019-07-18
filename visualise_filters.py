from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

with open('model.pickle', 'rb') as input_file:
    model = pickle.load(input_file)

for i in range(len(model.layers)):
    layer = model.layers[i]

    if not layer.name.startswith('conv2d'):
        continue

    print(i, layer.name, layer.output.shape)

model = Model(inputs = model.inputs, output = model.layers[1].output)

img = cv2.imread('small_test_images/3274050.png', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img = img_to_array(img)
#img = expand_dims(img, axis = 0)
img = np.array(img)
img = img.reshape(-1, 400, 400, 3)

feature_maps = model.predict([img, np.array([0.98])])

ix = 1
for _ in range(5):
    for _ in range(4):
        ax = plt.subplot(5, 4, ix)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(feature_maps[0, :,:,ix-1], cmap='rainbow')
        ix += 1
plt.show()
