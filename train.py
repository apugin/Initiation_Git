import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import numpy as np
from data import data_set
from callback import les_callbacks
from model import build_network
def train(lr=0.01, batchsize=64,epochs=5):
    data,label  = data_set()
    Model=build_network()
    Model.compile(
    optimizer='adam',
    loss='binary_crossentropy')
    # metrics=skimage.metrics.peak_signal_noise_ratio(image_true, image_test, *, data_range=None))
    Model.fit(data,label, batch_size=batchsize, epochs=epochs, callbacks=les_callbacks)


    