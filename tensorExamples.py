import requests, pandas, io, numpy, argparse, math
import numpy as np
import featureEngineering as fe
from myutils import *
from mpmath import *
from gdsolvers import *
import logging as log
import inspect
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import shuffle
import tensorflow as tf

def test_basic_tensor():
    x = tf.Variable(3, name='x')
    y = tf.Variable(4, name='y')
    f = x*x*y + y + 2

    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    log.info(result)
    sess.close()
    return result

def test_basic_tensor2():
    x = tf.Variable(3, name='x')
    y = tf.Variable(4, name='y')
    f = x*x*y + y + 2

    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()
        log.info(result)
    return result

def test_basic_tensor3():
    tf.reset_default_graph()
    x = tf.Variable(3, name='x')
    y = tf.Variable(4, name='y')
    f = x*x*y + y + 2

#    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
#        init.run()  # init variables
        result = f.eval()
        log.info(result)
    return result

def test_basic_tensor4():
    w = tf.constant(3)
    x = w + 2  # 5
    y = x + 5  # 10 
    z = y + 10 # 20
    with tf.Session():
        log.info(y.eval())
        log.info(z.eval())
    # or in batch y,z = sess.run([y,z])

def test_linreg_tensor():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()  # auto-caches to /Users/dougfoo/scikit_learn_data/
    m,n = housing.data.shape   # 20640,8 shape
    housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]
    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
    y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')

    X = tf.constant(np.array([[1,4],[1,5],[1,6]]), dtype=tf.float32, name='X')
    y = tf.constant(np.array([[8],[10],[12]]), dtype=tf.float32, name='y')

    XT = tf.transpose(X)  
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y) #not very natural..for (X^T*T)-1 * X^T*y 
    with tf.Session():
        theta_value = theta.eval()
        print (theta_value)
    return theta_value

if __name__ == "__main__":
    log.getLogger().setLevel(log.INFO)
    test_basic_tensor()
    test_basic_tensor2()
    test_basic_tensor3()
    test_basic_tensor4()
    test_linreg_tensor()
