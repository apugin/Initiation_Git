import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
def train(Model,data,labels,batch_size=64,epochs=5):
    Model.fit(data,labels, batch_size=64, epochs =5)
