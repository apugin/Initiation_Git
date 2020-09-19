import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, UpSampling
import numpy as np
def build_network(resolution=(32,32,3),new_resolution =(60,60,3)):
    #Input_shape = (resolution[0]*resolution[1],)
    Model = Sequential()
    Model.add(UpSampling2D((2,2), input_shape = (32,32,3,)))
    Model.add(Conv2D(filters= 32, kernel_size =(5,5) , padding = 'same'))
    Model.add(Conv2D(filters= 3, kernel_size =(4,4) , padding = 'same'))
    return(Model)
