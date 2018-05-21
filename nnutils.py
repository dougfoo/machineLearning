import tensorflow as tf
import numpy as np
import pandas as pd
import urllib.request as request
import matplotlib.pyplot as plt

# 1 hot encode a series
def encode(series):
  return pd.get_dummies(series.astype(str)) # ?


def create_train_model(hidden_nodes, num_iters, Xtrain, ytrain):
    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(120, 4), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(120, 3), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(4, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 3), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    loss_plot = []
    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: Xtrain, y: ytrain})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: Xtrain, y: ytrain}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2

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

def neuron_layer_eager(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])  # 1 or 0 - training rows
        stddev = 2.0 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)  # tensor n_input x n_neuron ?
        W = tf.contrib.eager.Variable(init, name="kernel")  # tensor [MxN] ?
        b = tf.contrib.eager.Variable(tf.zeros([n_neurons]),name='bias')  # row [1xN] ?
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
