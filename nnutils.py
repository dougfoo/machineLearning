import tensorflow as tf
import numpy as np
import pandas as pd
import urllib.request as request
import matplotlib.pyplot as plt
from myutils import *

# 1 hot encode a series
def encode(series):
  return pd.get_dummies(series.astype(str)) # ?

#######################
# tensorflow specific #
#######################

# for plotting weight summaries
def tf_var_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# adapted build layers, credit:  # from http://stackabuse.com/tensorflow-neural-network-tutorial/
def create_train_model(hidden_nodes, num_iters, Xtrain, ytrain, step_size=0.005):
    tf.reset_default_graph()

    # Placeholders for input and output data
    with tf.name_scope("InputsXY"):
        X = tf.placeholder(shape=Xtrain.shape, dtype=tf.float64, name='X')   # 120,4
        y = tf.placeholder(shape=ytrain.shape, dtype=tf.float64, name='y')   # 120,3

    # Variables for two group of weights between the three layers of the network
    # np.random.seed(rseed)   # depends if you want repeatability or not
    with tf.name_scope("Weights"):
        np.random.seed(0)
        W1 = tf.Variable(np.random.rand(Xtrain.shape[1], hidden_nodes), dtype=tf.float64, name='W1')  #4,n
        W2 = tf.Variable(np.random.rand(hidden_nodes, ytrain.shape[1]), dtype=tf.float64, name='W2')  #n,3
        tf_var_summaries(W1)
        tf_var_summaries(W2)

    # Create the neural net graph
    with tf.name_scope("fwdeval"):
        A1 = tf.sigmoid(tf.matmul(X, W1))
        y_est = tf.sigmoid(tf.matmul(A1, W2))
        tf_var_summaries(y_est)

    # Define a loss function
    with tf.name_scope("lossf"):
        deltas = tf.square(y_est - y)
        loss = tf.reduce_sum(deltas)
        tf.summary.scalar('log_loss', loss)       # logs are coming wrong log_loss[_1....n]

    # Define a train operation to minimize the loss (this is bit opaque)
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(step_size)
        train = optimizer.minimize(loss) 

    # Initialize variables and run session
    file_writer = tf.summary.FileWriter(getLogDir(), tf.get_default_graph())
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(num_iters):
            _,l,w1,w2 = sess.run([train,loss,W1,W2], feed_dict={X: Xtrain, y: ytrain})
            if (i % 200 == 0):
                print ('gd-nodes: %d, iteration %d, loss: %f'%(hidden_nodes,i,l))
                merged = tf.summary.merge_all() 
                summary = sess.run(merged, feed_dict={X: Xtrain, y: ytrain})
                file_writer.add_summary(summary, i)
        print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, l))
        file_writer.close()
        sess.close()

    return w1, w2

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

def relu(X):
    with tf.name_scope("relu"):
        with tf.variable_scope("relu", reuse=True):
            threshold = tf.get_variable("threshold")
            w_shape = (int(X.get_shape()[1]),1)
            w = tf.Variable(tf.random_normal(w_shape), name='weights')
            b = tf.Variable(0.0, name='bias')
            z = tf.add(tf.matmul(X,w), b, name='z')
            return tf.maximum(z, threshold,name='max')


####################
# pytorch specific #
####################
import torch
import torch.nn as nn
import torch.nn.functional as tfun

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # forward prop
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = tfun.max_pool2d(tfun.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = tfun.max_pool2d(tfun.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  #purpose?
        x = tfun.relu(self.fc1(x))
        x = tfun.relu(self.fc2(x))
        x = self.fc3(x)  # linear transform
        return x

    # whats purpose here
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# extends Net w/ fc internal model 500:20:2 model, using sigmoid
class GagaNet(nn.Module):
    def __init__(self):
        super(GagaNet, self).__init__()
        self.inp = nn.Linear(500, 20)
        self.hid = nn.Linear(20, 2)

    # forward prop
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.inp(x)
        x = tfun.sigmoid(self.inp(x))
        x = self.hid(x)
        x = tfun.sigmoid(self.hid(x))
        return x
