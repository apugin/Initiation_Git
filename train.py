from data import data_set
from callback import make_callback
from model import build_network
import tensorflow as tf


def train(lr=0.01, batchsize=64, epochs=5, ckpt_dir="./checkpoint", log_dir="log_dir"):
    data, label = data_set(phase="train")
    Model = build_network()
    Model.compile(optimizer=tf.keras.optimizers.Adam(), loss='MSE')
    # metrics= skimage.metrics.peak_signal_noise_ratio(image_true, image_test, *, data_range=None))
    callbacks = make_callback(ckpt_dir,log_dir)
    Model.fit(data, label, batch_size=batchsize, epochs=epochs, callbacks=callbacks)

#dire à Alex, de passer le ckpt_dir en argde train dans main et le log_dir

