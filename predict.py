import cv2
import numpy as np

import pickle

# open model file
with open('model.pickle', 'rb') as input_file:
    model = pickle.load(input_file)


test_img = cv2.imread('output/012866.png', 1)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
test_img = np.array(test_img)
# test_img = test_img.flatten()
test_input = test_img.reshape(-1, 940, 940, 3)
print(test_input.shape)

results = model.predict_on_batch([test_input,  np.array([14.0233432])])

print(results)
