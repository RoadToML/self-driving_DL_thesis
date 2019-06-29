import cv2
import numpy as np
import pandas as pd

import pickle

# open model file
with open('model.pickle', 'rb') as input_file:
    model = pickle.load(input_file)

df = pd.read_csv('test_labels.csv',\
names = ['image', 'velocity', 'steering_angle', 'outcome'],\
converters = {'image': lambda x: str(x),\
              'steering_angle': lambda x: round(float(x)/70, 5)})

for img, velocity, angle in zip(df['image'], df['velocity'], df['steering_angle']):

    try:
        test_img = cv2.imread(f'test_images/{img}.png', 1)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = np.array(test_img)
        # test_img = test_img.flatten()
        test_input = test_img.reshape(-1, 940, 940, 3)

        results = model.predict([test_input,  np.array([velocity])])

        print(img, results, angle)

    except cv2.error:
        pass
