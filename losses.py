import keras.backend as K
import tensorflow as tf


def binary_entropy_drop(y_true, y_pred):
    drop_rate = 0.0
    return K.sum(K.dropout(K.binary_crossentropy(y_true, y_pred), drop_rate), axis=-1) #/ (1 - drop_rate + K.epsilon())


def binary_entropy_class_drop(y_true, y_pred):
    drop_rate = 0.0
    # mask_loss = tf.stack(tf.reduce_max(y_true, axis=-1) * y_true.shape[-1], axis=-1)
    mask_loss = tf.expand_dims(tf.reduce_max(y_true, axis=-1), -1)
    return K.sum(K.mean(mask_loss * K.binary_crossentropy(y_true, y_pred), axis=-1), axis=-1) #/ (1 - drop_rate + K.epsilon())
