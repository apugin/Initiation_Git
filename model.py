import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
def build_network(resolution=(32,32,3),new_resolution =(60,60,3)):
    #Input_shape = (resolution[0]*resolution[1],)
    Model = Sequential()
    Model.add(Dense(new_resolution[0]*new_resolution[1]*new_resolution[2], activation ='relu'))
    Model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
    return(Model)
