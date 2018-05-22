import requests, pandas, io, numpy, argparse, math, inspect
import numpy as np
import matplotlib.pyplot as plt
import featureEngineering as fe
import logging as log
import tensorflow as tf
from nnutils import *
from myutils import *
from mpmath import *
from gdsolvers import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import shuffle

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

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
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

# not complete yet - 
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
                summary_str = ll_summary.eval()  
                step = epoch
                file_writer.add_summary(summary_str, step)
            sess.run(training_op)   # whats an opp
        best_theta = theta.eval()
    print(best_theta)
    file_writer.close()
    return best_theta

def test_nn_tensor():
    tf.reset_default_graph()
    n_epocs = 10  #40
    batch_size = 50
    learning_rate = 0.01
 
    ## NN setup phase
    n_inputs = 28*28
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name='y')

    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(X, n_hidden1, "hidden1")
        hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, "outputs", )

#    with tf.name_scope("dnn"):
#        hidden1 = tf.layers.dense(X,n_hidden1, name="hidden1")
#        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
#        logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    ## exec run phase
    init = tf.global_variables_initializer()
    file_writer = tf.summary.FileWriter(getLogDir(),tf.get_default_graph())
    saver = tf.train.Saver()

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/")

    with tf.Session() as sess:
        init.run()
        for epoc in range(n_epocs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                print (X_batch.shape, y_batch.shape)
                print (y_batch)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval({X: mnist.validation.images, y: mnist.validation.labels})
            print (epoc, "train accuracy:", acc_train, "Val accuracy:", acc_val)
        save_path = saver.save(sess, "./tf_logs/my_model_final.ckpt")
#    file_writer.add_summary(summary_str, step)
    file_writer.close()

# neural network
# from http://stackabuse.com/tensorflow-neural-network-tutorial/
def test_nn2_tensor():
    tf.reset_default_graph()

    # Download dataset
    IRIS_TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
    IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

    names = ['sepal-length', 'sepal-width',
            'petal-length', 'petal-width', 'species']
    train = pd.read_csv(IRIS_TRAIN_URL, names=names, skiprows=1)
    test = pd.read_csv(IRIS_TEST_URL, names=names, skiprows=1)

    # Train and test input data
    Xtrain = train.drop("species", axis=1)
    Xtest = test.drop("species", axis=1)

    # Encode target values into binary ('one-hot' style) representation
    ytrain = pd.get_dummies(train.species)
    ytest = pd.get_dummies(test.species)

    print (Xtrain.shape, type(Xtrain))
    print (ytrain.shape, type(ytrain))

    # Plot the loss function over iterations
    num_hidden_nodes = [5, 10, 20]
    loss_plot = {5: [], 10: [], 20: []}
    weights1 = {5: None, 10: None, 20: None}
    weights2 = {5: None, 10: None, 20: None}
    num_iters = 500

    plt.figure(figsize=(12,8))  
    for hidden_nodes in num_hidden_nodes:  
        weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters, Xtrain, ytrain, loss_plot)
        plt.plot(range(num_iters), loss_plot[hidden_nodes], label="nn: 4-%d-3" % hidden_nodes)

    plt.xlabel('Iteration', fontsize=12)  
    plt.ylabel('Loss', fontsize=12)  
    plt.legend(fontsize=12)  

    # Evaluate models on the test set
    X = tf.placeholder(shape=(30, 4), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(30, 3), dtype=tf.float64, name='y')

    for hidden_nodes in num_hidden_nodes:
        # Forward propagation
        W1 = tf.Variable(weights1[hidden_nodes])
        W2 = tf.Variable(weights2[hidden_nodes])
        A1 = tf.sigmoid(tf.matmul(X, W1))
        y_est = tf.sigmoid(tf.matmul(A1, W2))

        # Calculate the predicted outputs
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_est, feed_dict={X: Xtest, y: ytest})

        # Calculate the prediction accuracy
        correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
            for estimate, target in zip(y_est_np, ytest.as_matrix())]
        accuracy = 100 * sum(correct) / len(correct)
        print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))
#    plt.show()


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
    # test_nn_tensor()
    test_nn2_tensor()

