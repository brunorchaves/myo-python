import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial.distance import pdist, squareform #scipy spatial distance
import sklearn as sk
import sklearn.metrics.pairwise
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LeakyReLU
from keras import metrics
from keras import backend as K
import time
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.utils import np_utils

#modified from https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
def recurrence_plot(s, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=10
    d = sk.metrics.pairwise.pairwise_distances(s)
    d = np.floor(d / eps)
    d[d > steps] = steps
    #Z = squareform(d)
    return d

# random_series = np.random.random(1000)
# ax = fig.add_subplot(1, 3, 1)
# ax.imshow(recurrence_plot(random_series[:,None]))

# sinus_series = np.sin(np.linspace(0,24,1000))
# ax = fig.add_subplot(1, 3, 2)
# ax.plot(sinus_series[:][:1000])

# ax = fig.add_subplot(1, 3, 3)
# ax.imshow(recurrence_plot(sinus_series[:,None]))
data = pd.read_csv("emg_Samples.csv",index_col=0)
data.drop(labels =["gesture"],axis=1,inplace=True)
dataTransposed = data.T
print(data.shape)



fig = plt.figure(figsize=(5,4))

x = np.arange(0, 1000) 
chanel0 = data.iloc[0].to_numpy()
chanel1 = data.iloc[1].to_numpy()
chanel2 = data.iloc[2].to_numpy()
chanel3 = data.iloc[3].to_numpy()
chanel4 = data.iloc[4].to_numpy()
chanel5 = data.iloc[5].to_numpy()
chanel6 = data.iloc[6].to_numpy()
chanel7 = data.iloc[7].to_numpy()
at = fig.add_subplot(8, 1, 1)
plt.plot(x,chanel0,'tab:blue')
at = fig.add_subplot(8, 1, 2)
plt.plot(x,chanel1,'tab:red')
at = fig.add_subplot(8, 1, 3)
plt.plot(x,chanel2,'tab:green')
at = fig.add_subplot(8, 1, 4)
plt.plot(x,chanel3,'tab:olive')
at = fig.add_subplot(8, 1, 5)
plt.plot(x,chanel4,'tab:purple')
at = fig.add_subplot(8, 1, 6)
plt.plot(x,chanel5,'tab:brown')
at = fig.add_subplot(8, 1, 7)
plt.plot(x,chanel6,'tab:cyan')
at = fig.add_subplot(8, 1, 8)
plt.plot(x,chanel7,'tab:pink')


fig2 = plt.figure(figsize=(5,4))
ax = fig2.add_subplot(1, 1, 1)
ax.imshow(recurrence_plot(dataTransposed,steps=1000))

plt.show()








