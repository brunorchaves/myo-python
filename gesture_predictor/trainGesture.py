import tensorflow as tf
import pandas as pd
# Setup plotting
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split


emgSamples =  pd.read_csv("emg_Samples.csv",index_col=0)
X = emgSamples.copy()
y = X.pop('gesture')
print(X.head())

# print(X)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.75)
print(X_train.shape, X_valid.shape)
print(y)

input_shape = (X.shape[1],)
print(input_shape)

model = keras.Sequential([
    # the hidden ReLU layers
    layers.BatchNormalization( input_shape= input_shape),
    layers.Dense(800, activation='relu'),
    layers.Dense(128, activation='relu'),
    # layers.Dense(units=512, activation='relu'),
    # the linear output layer 
    layers.Dense(5, activation= "softmax"),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs= 100)

test_loss, test_acc = model.evaluate(X_valid,y_valid)

print("Tested Acc:",test_acc)

