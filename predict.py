import cv2
import numpy as np
import pandas as pd

import pickle

predict = True

# open model file
with open('model.pickle', 'rb') as input_file:
    model = pickle.load(input_file)

df = pd.read_csv('test_labels.csv',\
names = ['image', 'velocity', 'steering_angle', 'outcome'],\
converters = {'image': lambda x: str(x),\
              'steering_angle': lambda x: round(float(x)/70, 5)})

df['normal_velocity'] = round(((df['velocity'] - min(df['velocity']))/ (max(df['velocity']) - min(df['velocity']))), 8)

evaluate_data = []
for img, velocity, angle in zip(df['image'], df['normal_velocity'], df['steering_angle']):

    try:
        test_img = cv2.imread(f'test_images/{img}.png', 1)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = np.array(test_img)
        # test_img = test_img.flatten()

        if predict:
            test_input = test_img.reshape(-1, 940, 940, 3)

            results = model.predict([test_input,  np.array([velocity])])

            print(img, results, angle)

        else:
            evaluate_data.append([test_img, velocity, angle])

    except cv2.error:
        pass

if not predict:
    test_X = [img for img, label, velocity in evaluate_data]
    test_Y = [label for img, label, velocity in evaluate_data]
    test_other_inp = [velocity for img, label, velocity in evaluate_data]

    test_X = np.array(test_X).reshape(-1, 940,940,3)
    test_Y = np.array(test_Y)
    test_other_inp = np.array(test_other_inp)

    print(model.evaluate([test_X, test_other_inp], test_Y))

    print(model.metrics)
