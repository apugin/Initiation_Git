import tensorflow as tf
#def Callback_pimp (keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
les_callbacks=tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints",
    save_weights_only=True,
    save_freq="epoch",
    )


