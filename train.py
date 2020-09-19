import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
def train(Model,data,label,batch_size=64,epochs=5):
    label = label.reshape((len(label), np.prod(label.shape[1:])))
    data = data.reshape((len(data), np.prod(data.shape[1:])))
    Model.fit(data,labels, batch_size=64, epochs =5)
    