from __future__ import print_function
import pandas as pd
import numpy as np 
from myo.utils import TimeInterval
import myo
import sys
from threading import Lock, Thread
from matplotlib import pyplot as plt
import myo
import numpy as np
from collections import deque

columns= 100
rows = 8
dimensions = (rows,columns)
signal_header = np.zeros(columns,dtype='object')


#fill the signal header with its names
for i in range(0, columns):
    if(i == columns-1):
        signal_header[i] = "gesture"
    else:
        signal_header[i]= "sample_ "+ str(i);


signal_array=np.zeros(dimensions)

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
    emg_data = np.array([x[1] for x in emg_data])
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
  
myo.init(bin_path=r'D:\Documentos\GitHub\myoPython\myo-sdk-win-0.9.0\bin')
hub = myo.Hub()
listener = EmgCollector(512)
with hub.run_in_background(listener.on_event):
    for i in range(1,512):
        data = Plot(listener).display()
        # print(data)
 
#creates the dataframe

df = pd.DataFrame(data=signal_array,  columns=signal_header)
print(data)
print(len(data[0]))
print(df.head())
plt.plot(data)
# df.to_csv('out.csv')