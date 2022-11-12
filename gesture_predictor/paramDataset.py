from __future__ import print_function
from pickle import FALSE
from signal import signal
from sqlite3 import Timestamp
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
clear = lambda: os.system('cls')
import time
from gestureClass import gesture

def current_milli_time():
    return round(time.time() * 1000)

#Infos :
# - 64 samples per second


#Todo
# 1- Show the passage of time in milliseconds -- done!
# 2- increase the total number of samples -- done!
# 3- print the quantity of samples taken in real time --done!
# 3- put a warning for changing from rest to gesture  -- done!
# 4- automate name genaration -> gesture_spock_iter_# -- done!
# 5- plot the dat in matlab
# 6- make a visual interface to help the user do the gesture properly 
# 7- make the the gesture random

#Initial defitions
millis = 0
samples =1500
columns= samples + 1
rows = 9
# totalSamples = samples
totalColumns = samples+1
dimensions = (rows,columns)
dimensions2 = (rows,columns-1)

arraySize = (samples)+1
signal_header = np.zeros((arraySize),dtype='object')
rowsHeader = np.zeros((rows),dtype='object')
time_header = np.zeros((samples),dtype='object')

#fill the signal header with its names
for i in range(0, totalColumns):
    if(i == totalColumns-1):
        signal_header[i] = "gesture"
    else:
        signal_header[i]= "sample_ "+ str(i);




data = []
#receives the signal from the emg, saves 100 samples from each plate on the array

class EmgCollector(myo.DeviceListener):
  # """
  # Collects EMG data in a queue with *n* maximum number of elements.
  # """

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

# https://stackoverflow.com/questions/17907213/choosing-random-integers-except-for-a-particular-number-for-python
spock_g = gesture()
rock_g = gesture()
ok_g = gesture()
thumbs_Up_g = gesture()
pointer_g = gesture()
# Take the type of gesture and sum
# take a number from 1 to 5
# see if the gesture chosen is full 
# if it is, drop the number and take other except the ones not fulled

labelArray =np.zeros(1)
takingSamples = 1
dimensions_f = (0,arraySize)
gestureArray=np.empty(dimensions_f)
quantSamples = 0 

class_names = ['Spock','Rock','Ok!','Thumbs Up','Pointer','Released']
gestureIndex = int(input("Enter type of gesture:"))
labelArray[0] = gestureIndex
iterNumber = input("Enter number of iteration:")

# time initial
timeStamp = current_milli_time();
sampleFreq = False
while(takingSamples == 1 ):
    quantSamples += 1
    print("Samples collection started, please rest the arm")
    myo.init(bin_path=r'D:\Documentos\GitHub\myoPython\myo-sdk-win-0.9.0\bin')
    hub = myo.Hub()
    listener = EmgCollector(samples)
    with hub.run_in_background(listener.on_event):
        for i in range(1,samples):
            millis = current_milli_time() - timeStamp
            time_header[i] = millis
            # Samples per second counting:
            if(millis>=1000 and sampleFreq == False):
              sampleFreq = True;
              print(str(i) + " samples por second")
            # Warning to tell when do the gesture:
            if(i < samples/2):
              clear()
              print(str(samples/2-i) + " samples left to start the gesture" )
            else:
              clear()
              print("Please do the gesture") 
              print(str(samples-i) + " samples left to finish the gesture" )

            data = Plot(listener).display()
            # print(data)

    # time in millis
    #concatenate signal
    signal_array=np.zeros(dimensions)
    signal_array[1:,:-1] = data
    channel_0 =  signal_array[0,:-1]
    channel_1 =  signal_array[1,:-1]
    channel_2 =  signal_array[2,:-1]
    channel_3 =  signal_array[3,:-1]
    channel_4 =  signal_array[4,:-1]
    channel_5 =  signal_array[5,:-1]
    channel_6 =  signal_array[6,:-1]
    channel_7 =  signal_array[7,:-1]
    signal_array[0,:-1] = time_header;
    # signal_array[1,:-1] = time_header;

    # arrayLine = np.concatenate((channel_0,channel_1, channel_2,channel_3,channel_4,channel_5,channel_6,channel_7,time_header,labelArray), axis=None);
    # gestureArray = np.vstack([arrayLine,gestureArray])# stack lines of signal
    # gestureArray =arrayLine;
    takingSamples  = int(input("samples caught : " + str(quantSamples) + " Continue taking samples?"))

print(signal_array)
print(signal_array.shape)

# name_csv = input('Give a name to the data frame: ')
name_csv = "gesture_"+class_names[gestureIndex]+"_iter_"+ str(iterNumber)
#creates the dataframe
df = pd.DataFrame(data=signal_array,  columns=signal_header )
# print(df)
#correct the datafram for the recurrence plot 
df.to_csv(name_csv)
df = pd.read_csv(name_csv,index_col=0)
# df.drop(labels =["gesture"],axis=1,inplace=True)
# dfTransposed = df.T
print(df)

