import tensorflow as tf
import pandas as pd
# Setup plotting
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from signal import signal
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

# Load the trained model
model = tf.keras.models.load_model("gesturePredictor_RNN.model")
print("model loaded")

# Using real time data
# Todo
# 1 capture data
# 2 put it into a moving window pandas array
# 3 make the std scale of the array
# 4 make a single prediction


# Capture data

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

data = []

samples =100
columns= samples 
rows = 8
totalSamples = samples*8
dimensions = (rows,columns)
arraySize = (samples*rows)+1
dimensions_f = (0,arraySize)
gestureArray=np.empty(dimensions_f)



print("collecting samples, please make the gesture")
myo.init(bin_path=r'D:\Documentos\GitHub\myoPython\myo-sdk-win-0.9.0\bin')
hub = myo.Hub()
listener = EmgCollector(samples)
with hub.run_in_background(listener.on_event):
    for i in range(1,samples):
        data = Plot(listener).display()
        
signal_array=np.zeros(dimensions)
signal_array[:,:] = data
scaler = StandardScaler()
signal_array_scaled = scaler.fit_transform(signal_array)

channel_0 =  signal_array_scaled[0,:]
channel_1 =  signal_array_scaled[1,:]
channel_2 =  signal_array_scaled[2,:]
channel_3 =  signal_array_scaled[3,:]
channel_4 =  signal_array_scaled[4,:]
channel_5 =  signal_array_scaled[5,:]
channel_6 =  signal_array_scaled[6,:]
channel_7 =  signal_array_scaled[7,:]



arrayLine = np.concatenate((channel_0,channel_1, channel_2,channel_3,channel_4,channel_5,channel_6,channel_7), axis=None);
Single_gesture = arrayLine.reshape(1,800)   # Shape conversion of the input data to the model input shape requisit

print(Single_gesture)
print("Single_gesture shape : " + str(Single_gesture.shape))
print("Single_gesture type : " + str(type(Single_gesture)))





prediction = model.predict(Single_gesture)
class_names = ['Spock','Rock','Ok!','Thumbs Up','Pointer']
print("Previsão: " + class_names[np.argmax(prediction[0])] )



# # Using dataset
# name_df = input("diga o nome do csv:")
# emgSamples =  pd.read_csv(name_df,index_col=0)
# X = emgSamples.copy()
# y = X.pop('gesture')
# X_scaled = scaler.fit_transform(X)

# X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, train_size=0.75)
# class_names = ['Spock','Rock','Ok!','Thumbs Up','Pointer']
# print(type(X_valid))                           # numpy.ndarray
# X_single_gesture = X_valid[0].reshape(1,800)   # Shape conversion of the input data to the model input shape requisit
# prediction = model.predict(X_single_gesture)

# y_v_array = np.array(y_valid);
# print("Previsão: " + class_names[np.argmax(prediction[0])] + "  Gesto efetuado: " + class_names[int(y_v_array[0])])


