import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
import data
def train(Model,label,batchsize=64,epochs=5):
    data = data_set()
    Model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    Model.fit(data,labels, batch_size=batchsize, epochs =epochs)
    