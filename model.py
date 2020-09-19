import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
def build_network(resolution=(40,40),new_resolution =(60,60)):
    #Input_shape = (resolution[0]*resolution[1],)
    Model = Sequential()
    Model.add(Flatten(input_shape = resolution))
    Model.add(Dense(new_resolution[0]*new_resolution[0], activation ='relu'))
    Model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    return(Model)
Model = build_network()