import tensorflow as tf
import pandas as pd
# Setup plotting
import matplotlib.pyplot as plt
import cv2
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

#Image display variables
plt.ion()
currentImg = " "
lastImg = " "
imdir = 'img/'
img = cv2.imread(imdir+'relaxedHand.jpg', cv2.IMREAD_COLOR)
cv2.imshow("relaxedHand", img)
#Real time classification
print("collecting samples, please make the gesture")
myo.init(bin_path=r'D:\Documentos\GitHub\myoPython\myo-sdk-win-0.9.0\bin')
hub = myo.Hub()
listener = EmgCollector(samples)
with hub.run_in_background(listener.on_event):
  while 1:
        for i in range(1,samples):
          data = Plot(listener).display()
          
        # print("Data shape: " + str(data.shape))
        signal_array=np.zeros(dimensions)
        signal_array[:,:] = data

        channel_0 =  signal_array[0,:]
        channel_1 =  signal_array[1,:]
        channel_2 =  signal_array[2,:]
        channel_3 =  signal_array[3,:]
        channel_4 =  signal_array[4,:]
        channel_5 =  signal_array[5,:]
        channel_6 =  signal_array[6,:]
        channel_7 =  signal_array[7,:]

        arrayLine = np.concatenate((channel_0,channel_1, channel_2,channel_3,channel_4,channel_5,channel_6,channel_7), axis=None);
        mean = arrayLine.mean(axis=0)
        std = arrayLine.std(axis=0)
        arrayLine -=mean
        arrayLine /=std
        Single_gesture = arrayLine.reshape(1,800)   # Shape conversion of the input data to the model input shape requisit


        # scaler = StandardScaler()
        # single_gesture_scaled = scaler.fit_transform(Single_gesture)
        # print(Single_gesture)
        # print("Single_gesture shape : " + str(Single_gesture.shape))
        # print("Single_gesture type : " + str(type(Single_gesture)))
        a = 0

        prediction = model.predict(Single_gesture)
        class_names = ['Spock','Rock','Ok!','Thumbs Up','Pointer','Released']
        print("Previsão: " + class_names[np.argmax(prediction[0])] )
        if((class_names[np.argmax(prediction[0])]) == "Spock"): 
          currentImg = "Spock"
          if(currentImg != lastImg):   
            cv2.destroyAllWindows()
            img = cv2.imread(imdir+'spockHand.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(300, 300))
            cv2.imshow("Spock", img)
        elif((class_names[np.argmax(prediction[0])]) == "Rock"):
          currentImg = "Rock"
          if(currentImg != lastImg): 
            cv2.destroyAllWindows()
            img = cv2.imread(imdir+'rockHand.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(300, 300))            
            cv2.imshow("Rock", img)
        elif((class_names[np.argmax(prediction[0])]) == "Ok!"):
          currentImg = "Ok"
          if(currentImg != lastImg): 
            cv2.destroyAllWindows()
            img = cv2.imread(imdir+'okHand.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(300, 300))            
            cv2.imshow("OK", img)
        elif((class_names[np.argmax(prediction[0])]) == "Thumbs Up"):
          currentImg = "Thumbs Up"
          if(currentImg != lastImg):
            cv2.destroyAllWindows()   
            img = cv2.imread(imdir+'thumbsUpHand.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(300, 300))            
            cv2.imshow("ThumbsUp", img)
        elif((class_names[np.argmax(prediction[0])]) == "Pointer"):
          currentImg = "Pointer"
          if(currentImg != lastImg): 
            cv2.destroyAllWindows() 
            img = cv2.imread(imdir+'pointerHand.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(300, 300))            
            cv2.imshow("Pointer", img)
        elif((class_names[np.argmax(prediction[0])]) == "Released"):
          currentImg = "Released"
          if(currentImg != lastImg):
            cv2.destroyAllWindows()   
            img = cv2.imread(imdir+'relaxedHand.jpg', cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(300, 300))            
            cv2.imshow("Released", img)
        cv2.waitKey(1)
        lastImg = currentImg


# scaler = StandardScaler()

# # Using dataset
# name_df = input("diga o nome do csv:")
# emgSamples =  pd.read_csv(name_df,index_col=0)
# X = emgSamples.copy()
# y = X.pop('gesture')
# # X_scaled = scaler.fit_transform(X)
# # print(X_scaled)

# X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75)
# class_names = ['Spock','Rock','Ok!','Thumbs Up','Pointer','Released']
# prediction = model.predict(X_valid)
# X_valid_np = X_valid.to_numpy()
# print(type(X_valid_np))
# print((X_valid.shape))

# for i in range(0,45):
#   single_gesture = X_valid_np[i].reshape(1,800)
#   print(single_gesture.shape)
#   prediction = model.predict(single_gesture)
#   y_v_array = np.array(y_valid);
#   print("Previsão: " + class_names[np.argmax(prediction[0])] + "  Gesto efetuado: " + class_names[int(y_v_array[i])])




