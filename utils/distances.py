import numpy as np
import pandas as pd
import tensorflow as tf

def cosine(a, b):
    """
    Function to obtain the cosine distance between two kinematics vectors for all events in the input

    Args:
        a (tensorflow.tensor): first matrix of events and kinematics
        b (tensorflow.tensor): second matrix of events and kinematics

    Returns:
        (float) cosine distance
    """
    numerator = tf.linalg.matmul(a, b, transpose_b=True)
    denominator = tf.math.sqrt(tf.tensordot(tf.math.reduce_sum(tf.math.square(a),axis=1), tf.math.reduce_sum(tf.math.square(b),axis=1), axes=0))
    print(1 - numerator / denominator)

    return 1 - numerator / denominator

def euclidean(a, b):
    """
    Function to obtain the euclidean distance between two kinematics vectors for all events in the input

    Args:
        a (tensorflow.tensor): first matrix of events and kinematics
        b (tensorflow.tensor): second matrix of events and kinematics

    Returns:
        (float) euclidean distance
    """
    a_expanded = tf.expand_dims(a, axis=1)
    b_expanded = tf.expand_dims(b, axis=0)

    return tf.sqrt(tf.reduce_sum(tf.square(a_expanded-b_expanded),axis=-1))

def cityblock(a, b):
    """
    Function to obtain the cityblock distance between two kinematics vectors for all events in the input

    Args:
        a (tensorflow.tensor): first matrix of events and kinematics
        b (tensorflow.tensor): second matrix of events and kinematics

    Returns:
        (float) cityblock distance
    """
    a_expanded = tf.expand_dims(a, axis=1)
    b_expanded = tf.expand_dims(b, axis=0)

    return tf.reduce_sum(tf.abs(a_expanded-b_expanded),axis=-1)
