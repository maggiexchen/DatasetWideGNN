import numpy as np
import pandas as pd
import tensorflow as tf

def cosine(a, b):
    numerator = tf.linalg.matmul(a, b, transpose_b=True)
    denominator = tf.math.sqrt(tf.tensordot(tf.math.reduce_sum(tf.math.square(a),axis=1), tf.math.reduce_sum(tf.math.square(b),axis=1), axes=0))
    print(1 - numerator / denominator)
    return 1 - numerator / denominator

def euclidean(a, b):
    a_expanded = tf.expand_dims(a, axis=1)
    b_expanded = tf.expand_dims(b, axis=0)
    return tf.sqrt(tf.reduce_sum(tf.square(a_expanded-b_expanded),axis=-1))

def cityblock(a, b):
    a_expanded = tf.expand_dims(a, axis=1)
    b_expanded = tf.expand_dims(b, axis=0)
    return tf.reduce_sum(tf.abs(a_expanded-b_expanded),axis=-1)

