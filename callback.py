import tensorflow as tf


def make_callback(ckpt_dir, log_dir="logs"):

    les_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_dir+"/epoch_{epoch:02d}",
                save_weights_only=True,
                save_freq="epoch",
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch",
                profile_batch=2,
                embeddings_freq=0,
                embeddings_metadata=None
            )
        ]

    return les_callbacks

