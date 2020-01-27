import numpy as np
import keras.backend as K
import tensorflow as tf

def recall(y_true, y_pred):
    a = y_true >= 0.5
    b = y_pred >= 0.5
    a = K.cast(a, K.floatx())
    b = K.cast(b, K.floatx())
    c = a * b
    return K.sum(c) / K.sum(a)

def recall_diag(y_true, y_pred):
    a = y_true >= 0.5
    b = y_pred >= 0.5
    a = K.cast(a, K.floatx())
    b = K.cast(b, K.floatx())
    c = a * b
    return K.sum(c) / K.sum(a)

def precision(y_true, y_pred):
    a = y_true >= 0.5
    b = y_pred >= 0.5
    a = K.cast(a, K.floatx())
    b = K.cast(b, K.floatx())
    c = a * b
    return K.sum(c) / (K.sum(b) + 0.0001)

def fmeasure(y_true, y_pred):
    r = recall(y_true, y_pred)
    p = precision(y_true, y_pred)

    f = 2.0 * r * p / (r + p + 0.0001)
    return f

def precision_diag(y_true, y_pred):
    a = y_true >= 0.5
    b = y_pred >= 0.5
    a = K.cast(a, K.floatx())
    b = K.cast(b, K.floatx())
    mask = tf.expand_dims(tf.reduce_max(y_true, axis=-1), -1)

    c = a * b
    return K.sum(c) / (K.sum(b * mask) + 0.01)

def recall_np(y_true, y_pred):
    a = np.array(y_true >= 0.5).astype(float)
    b = np.array(y_pred >= 0.5).astype(float)

    c = a * b
    return np.sum(c) / np.sum(a)

def precision_np(y_true, y_pred):
    a = np.array(y_true >= 0.5).astype(float)
    b = np.array(y_pred >= 0.5).astype(float)

    c = a * b
    return np.sum(c) / (np.sum(b) + 1.0)

