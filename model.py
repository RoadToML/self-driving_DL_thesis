from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense

import keras
import pandas as pd 
import cv2
import matplotlib.pyplot as plt

import os 
import time

# #######################################################
# ######## LOAD Dataframe   #############################
def load_data(csv_file):

    df = pd.read_csv(csv_file,\
    names = ['image', 'velocity', 'steering_angle', 'outcome'],\
    converters = {'image': lambda x: str(x)})

    print (df.head()) # DEBUG

    return df

# #######################################################
# ######## LOAD IMAGE DATA  #############################
def load_image_array(image_df):

    for i in image_df:

        image = cv2.imread(f'output/{i}.png', 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yield image

# #######################################################
# ######## PLOT IMAGE  ###################################      

def plot_image(image_arr):

    plt.imshow(image_arr)
    plt.show()

if __name__ == '__main__':


    df = load_data('velocity_labels.csv')

    for image in load_image_array(df['image']):

        plot_image(image)
        break



