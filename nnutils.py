import tensorflow as tf
import numpy as np
import pandas as pd
import urllib.request as request
import matplotlib.pyplot as plt
from myutils import *

# 1 hot encode a series
def encode(series):
  return pd.get_dummies(series.astype(str)) # ?

# adapted build layers, credit:  # from http://stackabuse.com/tensorflow-neural-network-tutorial/
def create_train_model(hidden_nodes, num_iters, Xtrain, ytrain, loss_plot, step_size=0.005):
    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=Xtrain.shape, dtype=tf.float64, name='X')   # 120,4
    y = tf.placeholder(shape=ytrain.shape, dtype=tf.float64, name='y')   # 120,3

    # Variables for two group of weights between the three layers of the network
    np.random.seed(0)
    W1 = tf.Variable(np.random.rand(Xtrain.shape[1], hidden_nodes), dtype=tf.float64)  #4,n
    W2 = tf.Variable(np.random.rand(hidden_nodes, ytrain.shape[1]), dtype=tf.float64)  #n,3

    # Create the neural net graph
    with tf.name_scope("input"):
        A1 = tf.sigmoid(tf.matmul(X, W1))
        y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    with tf.name_scope("lossf"):
        deltas = tf.square(y_est - y)
        loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(step_size)
        train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    file_writer = tf.summary.FileWriter(getLogDir(), tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)

        # Go through num_iters iterations
        for i in range(num_iters):
            sess.run(train, feed_dict={X: Xtrain, y: ytrain})
            l = sess.run(loss, feed_dict={X: Xtrain, y: ytrain})
            loss_plot[hidden_nodes].append(l)
            weights1 = sess.run(W1)
            weights2 = sess.run(W2)
# TODO figure out how to get this summary logging working
#            ll_summary = tf.summary.scalar('log_loss', l)
#            file_writer.add_summary(ll_summary)

        print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
        file_writer.close()
        sess.close()


    return weights1, weights2

# alt layer builder
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

# alt eager eval builder
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
