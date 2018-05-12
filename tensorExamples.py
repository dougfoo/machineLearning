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

def getLogDir():
    from datetime import datetime
    now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    root_logdir = 'tf_logs'
    logdir = "{}/run-{}/".format(root_logdir, now)
    return logdir

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

def test_grad_tensor_logging():
    tf.reset_default_graph()
    n_epochs = 400
    learning_rate = 0.01

    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()  # auto-caches to /Users/dougfoo/scikit_learn_data/
    m,n = housing.data.shape   # 20640,8 shape
    housing_data_plus_bias = np.c_[np.ones((m,1)),housing.data]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(housing.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
#    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')   #otherwise need super small learning rate
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")

#    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    theta = tf.Variable(tf.constant([[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0],[1.0]]), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')

    with tf.name_scope("loss"):
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name='mse')

    with tf.name_scope("gradients"):
        gradients = 2.0/m * tf.matmul(tf.transpose(X), error)
        training_op = tf.assign(theta, theta - learning_rate * gradients)   # what is this

    mse_summary = tf.summary.scalar('MSE',mse)

    file_writer = tf.summary.FileWriter(getLogDir(),tf.get_default_graph())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if (epoch % 25 == 0):
                print('Epoch %s MSE %s'%(epoch, mse.eval()))
                summary_str = mse_summary.eval()  # bug
                step = epoch
                file_writer.add_summary(summary_str, step)
            sess.run(training_op)   # whats an opp
        best_theta = theta.eval()
    print(best_theta)
    file_writer.close()
    return best_theta

def test_mod_tensor():
    tf.reset_default_graph()
    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None,n_features), name="X")
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold",shape=(),initializer=tf.constant_initializer(0.0))
    relus = [relu(X) for i in range(5)]
    output = tf.add_n(relus, name="output")
    file_writer = tf.summary.FileWriter(getLogDir(),tf.get_default_graph())

    init = tf.global_variables_initializer()

    # with tf.Session() as sess:
    #     sess.run(init)
    #     print (output.eval())

    file_writer.close()

def relu(X):
    with tf.name_scope("relu"):
        with tf.variable_scope("relu", reuse=True):
            threshold = tf.get_variable("threshold")
            w_shape = (int(X.get_shape()[1]),1)
            w = tf.Variable(tf.random_normal(w_shape), name='weights')
            b = tf.Variable(0.0, name='bias')
            z = tf.add(tf.matmul(X,w), b, name='z')
            return tf.maximum(z, threshold,name='max')

def test_logreg_tensor():
    tf.reset_default_graph()
    n_epochs = 400
    learning_rate = 0.01

    xs = np.array([[10,1],[11,2],[1,6]])
    ys = np.array([[1],[1],[0]])
    m = len(ys)
    X = tf.constant(xs, dtype=tf.float32, name='X')
    y = tf.constant(ys, dtype=tf.float32, name='y')

    theta = tf.Variable(tf.constant([[0.1],[0.1]]), name='theta')
    y_pred = tf.sigmoid(tf.matmul(X, theta, name='predictions'))

    with tf.name_scope("loss"):
        error = y_pred - y  # vs ll ?
        ll = tf.reduce_mean(tf.losses.log_loss(y,y_pred), name='log_loss')  # -log(x) or -log(1-x) ....

    with tf.name_scope("gradients"):
        gradients = 2.0/m * tf.matmul(tf.transpose(X), error)  # vs ll vs error 
        training_op = tf.assign(theta, theta - learning_rate * gradients)   # what is this

    ll_summary = tf.summary.scalar('log_loss',ll)

    file_writer = tf.summary.FileWriter(getLogDir(),tf.get_default_graph())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if (epoch % 25 == 0):
                print('Epoch %s Log_Loss %s'%(epoch, ll.eval()))
                summary_str = ll_summary.eval()  # bug
                step = epoch
                file_writer.add_summary(summary_str, step)
            sess.run(training_op)   # whats an opp
        best_theta = theta.eval()
    print(best_theta)
    file_writer.close()
    return best_theta

def test_gaga_tensor():
    tf.reset_default_graph()
    n_epochs = 400
    learning_rate = 0.01

    xMatrix,yArr,features,fnames = fe.getGagaData(maxrows=500,stopwords='english')

    xMatrix = shuffle(xMatrix, random_state=0)   
    yArr = shuffle(yArr, random_state=0) 

    partition = int(.70*len(yArr))

    trainingMatrix = xMatrix[:partition]
    trainingY = yArr[:partition]
    testMatrix = xMatrix[partition:]
    testY = yArr[partition:]

    X = np.array(trainingMatrix)
    Y = trainingY
    from logisticRegression import reduceFeatures
    X,rfeatures = reduceFeatures(X, Y, features, 500)

    # convert to TensorFlow formats
    xs = X
    ys = np.array(Y).reshape(-1,1)  #col orient
#    np.random.seed(42)
#    guesses = np.random.rand(1,len(xs)).astype('float32') 
    guesses = np.array([0.01]*len(xs[0]),dtype='float32' ).reshape(-1,1)  # col orient
    m = len(ys)
    X = tf.constant(xs, dtype=tf.float32, name='X')
    y = tf.constant(ys, dtype=tf.float32, name='y')

    theta = tf.Variable(tf.constant(guesses), name='theta')
    y_pred = tf.sigmoid(tf.matmul(X, theta, name='predictions'))

    with tf.name_scope("loss"):
        error = y_pred - y  # vs ll ?
        ll = tf.reduce_mean(tf.losses.log_loss(y,y_pred), name='log_loss')  # -log(x) or -log(1-x) ....

    with tf.name_scope("gradients"):
        gradients = 2.0/m * tf.matmul(tf.transpose(X), error)  # vs ll vs error 
        training_op = tf.assign(theta, theta - learning_rate * gradients)   # what is this

    ll_summary = tf.summary.scalar('log_loss',ll)

    file_writer = tf.summary.FileWriter(getLogDir(),tf.get_default_graph())

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if (epoch % 25 == 0):
                print('Epoch %s Log_Loss %s'%(epoch, ll.eval()))
                summary_str = ll_summary.eval()  # bug
                step = epoch
                file_writer.add_summary(summary_str, step)
            sess.run(training_op)   # whats an opp
        best_theta = theta.eval()
    print(gf(best_theta))   # scores should be similar to sckit and grad5 solver
    print([str(x) for x in rfeatures[:20]])
    file_writer.close()

    # reduce test set similarly (note below works because we know full set of train+test features ahead of time)    
    df = pandas.DataFrame(testMatrix, columns=features)   # new df w/ column names
    X = df[rfeatures].as_matrix()                # filter out only rfeatures

    testRes = np.dot(X, best_theta)
    testResRound = [round(sigmoid(x),0) for x in testRes]
    testDiffs = np.array(testResRound) - np.array(testY)
    log.warn ('raw results %s '%(gf(testRes)))
    log.warn ('sig results %s'%gf([sigmoid(x) for x in testRes]))
    log.warn ('0|1 results %s'%([round(sigmoid(x)) for x in testRes]))
    log.warn (testDiffs)
    log.error ('mymodel errors: %s / %s = %f'%(sum([abs(x) for x in testDiffs]),len(testY),sum([abs(x) for x in testDiffs])/len(testY)))

    return best_theta

if __name__ == "__main__":
    log.getLogger().setLevel(log.INFO)
    # test_basic_tensor()
    # test_basic_tensor2()
    # test_basic_tensor3()
    # test_basic_tensor4()
    # test_linreg_tensor()
    # test_grad_tensor_logging()
    # test_mod_tensor()
    # test_logreg_tensor()
    test_gaga_tensor()

