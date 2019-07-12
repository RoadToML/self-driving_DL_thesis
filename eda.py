import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('test_labels.csv',\
names = ['image', 'velocity', 'steering_angle', 'outcome'],\
converters = {'image': lambda x: str(x),\
              'steering_angle': lambda x: round(float(x)/70, 5)})

df['normal_velocity'] = round(((df['velocity'] - min(df['velocity']))/ (max(df['velocity']) - min(df['velocity']))), 8)

sns.distplot(df['normal_velocity'])

plt.show()
