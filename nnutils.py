import tensorflow as tf
import numpy as np

# X is MxN
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])  # 1 or 0 - training rows
        stddev = 2.0 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)  # tensor n_input x n_neuron ?
        W = tf.Variable(init, name="kernel")  # tensor [MxN] ?
        b = tf.Variable(tf.zeros([n_neurons]),name='bias')  # row [1xN] ?
        Z = tf.matmul(X,W) + b  # what is the shape result??  row [1xN] ?
        if (activation is not None):
            return activation(Z)
        else:
                return Z

def relu(X):
    with tf.name_scope("relu"):
        with tf.variable_scope("relu", reuse=True):
            threshold = tf.get_variable("threshold")
            w_shape = (int(X.get_shape()[1]),1)
            w = tf.Variable(tf.random_normal(w_shape), name='weights')
            b = tf.Variable(0.0, name='bias')
            z = tf.add(tf.matmul(X,w), b, name='z')
            return tf.maximum(z, threshold,name='max')
