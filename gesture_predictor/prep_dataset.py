from __future__ import print_function
from signal import signal
import pandas as pd
import numpy as np
from tomlkit import boolean 
from myo.utils import TimeInterval
import myo
import sys
from threading import Lock, Thread
from matplotlib import pyplot as plt
import myo
import numpy as np
from collections import deque
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch


from scipy.spatial.distance import pdist, squareform #scipy spatial distance
import sklearn as sk
import sklearn.metrics.pairwise
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


#Recurrence plot function
def recurrence_plot(s, eps=None, steps=None):
    if eps==None: eps=0.1
    if steps==None: steps=10
    d = sk.metrics.pairwise.pairwise_distances(s)
    d = np.floor(d / eps)
    d[d > steps] = steps
    #Z = squareform(d)cd
    return d


#Initial defitions
samples =100
columns= samples + 1
rows = 8
totalSamples = samples*8
totalColumns = totalSamples+1
dimensions = (rows,columns)
dimensions2 = (rows,columns-1)

signal_header = np.zeros(801,dtype='object')


#fill the signal header with its names
for i in range(0, totalColumns):
    if(i == totalColumns-1):
        signal_header[i] = "gesture"
    else:
        signal_header[i]= "sample_ "+ str(i);


data = []
#receives the signal from the emg, saves 100 samples from each plate on the array

class EmgCollector(myo.DeviceListener):
  """
  Collects EMG data in a queue with *n* maximum number of elements.
  """

  def __init__(self, n):
    self.n = n
    self.lock = Lock()
    self.emg_data_queue = deque(maxlen=n)

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  # myo.DeviceListener

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    with self.lock:
      self.emg_data_queue.append((event.timestamp, event.emg))


class Plot(object):

  def __init__(self, listener):
    self.n = listener.n
    self.listener = listener
    # self.fig = plt.figure()
    # self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
    # [(ax.set_ylim([-100, 100])) for ax in self.axes]
    # self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
    # plt.ion()

  def update_plot(self):
    emg_data = self.listener.get_emg_data()
    emg_data = np.array([x[1] for x in emg_data]).T
    return emg_data
    # for g, data in zip(self.graphs, emg_data):
    #   if len(data) < self.n:
    #     # Fill the left side with zeroes.
    #     data = np.concatenate([np.zeros(self.n - len(data)), data])
    #   g.set_ydata(data)
    # plt.draw()

  def display(self):
    data_local = self.update_plot()
    plt.pause(1.0 / 100)
    return data_local
  

labelArray =np.zeros(1)
takingSamples = 1
dimensions_f = (0,801)
gestureArray=np.empty(dimensions_f)

while(takingSamples == 1 ):
    labelArray[0] = input("Enter type of gesture:")
    print("collecting samples, please make the gesture")

    myo.init(bin_path=r'D:\Documentos\GitHub\myoPython\myo-sdk-win-0.9.0\bin')
    hub = myo.Hub()
    listener = EmgCollector(samples)
    with hub.run_in_background(listener.on_event):
        for i in range(1,samples):
            data = Plot(listener).display()
            # print(data)
    
    #concatenate signal
    signal_array=np.zeros(dimensions)
    signal_array[:,:-1] = data
    channel_0 =  signal_array[0,:-1]
    channel_1 =  signal_array[1,:-1]
    channel_2 =  signal_array[2,:-1]
    channel_3 =  signal_array[3,:-1]
    channel_4 =  signal_array[4,:-1]
    channel_5 =  signal_array[5,:-1]
    channel_6 =  signal_array[6,:-1]
    channel_7 =  signal_array[7,:-1]
    arrayLine = np.concatenate((channel_0,channel_1, channel_2,channel_3,channel_4,channel_5,channel_6,channel_7,labelArray), axis=None);
    gestureArray = np.vstack([arrayLine,gestureArray])
    takingSamples  = int(input("Continue taking samples?"))


#creates the dataframe
df = pd.DataFrame(data=gestureArray,  columns=signal_header)
# print(df)
#correct the datafram for the recurrence plot
df.to_csv('emg_Samples.csv')
df = pd.read_csv("emg_Samples.csv",index_col=0)
# df.drop(labels =["gesture"],axis=1,inplace=True)
# dfTransposed = df.T
print(df)


# #Plot the data
# fig, axs = plt.subplots(8)
# #Plot data
# for i in range(0,samples):
#   axs[0].plot(data[0][:samples],'tab:blue')
#   axs[1].plot(data[1][:samples],'tab:red')
#   axs[2].plot(data[2][:samples],'tab:green')
#   axs[3].plot(data[3][:samples],'tab:olive')
#   axs[4].plot(data[4][:samples],'tab:purple')
#   axs[5].plot(data[5][:samples],'tab:brown')
#   axs[6].plot(data[6][:samples],'tab:cyan')
#   axs[7].plot(data[7][:samples],'tab:pink')
# plt.show()

# #Plot recurrence plot
# fig2 = plt.figure(figsize=(5,4))
# ax = fig2.add_subplot(1, 1, 1)
# ax.imshow(recurrence_plot(dfTransposed,steps=1000))
# plt.show()