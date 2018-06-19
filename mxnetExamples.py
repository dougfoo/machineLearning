import mxnet as mx
import numpy as np
from myutils import *
from nnutils import *

def test_tutorial():
    fname = mx.test_utils.download('https://s3.us-east-2.amazonaws.com/mxnet-public/letter_recognition/letter-recognition.data')
    data = np.genfromtxt(fname, delimiter=',')[:,1:]
    label = np.array([ord(l.split(',')[0])-ord('A') for l in open(fname, 'r')])

    batch_size = 32
    ntrain = int(data.shape[0]*0.8)
    train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)

    print ('data.shape', data.shape)

    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
    net = mx.sym.Activation(net, name='relu1', act_type="relu")
    net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    print (net)

    mx.viz.plot_network(net)

    mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)  #alloc mem
    mod.init_params(initializer=mx.init.Uniform(scale=.1))  #init random vals
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))  
    metric = mx.metric.Accuracy()  # alias is create('acc')

    for epoch in range(5):
        train_iter.reset()
        metric.reset()
        for batch in train_iter:
            mod.forward(batch, is_train=True)       # compute predictions
            mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            mod.backward()                          # compute gradients
            mod.update()                            # update parameters
        print('Epoch %d, Training %s' % (epoch, metric.get()))

    import logging
    logging.getLogger().setLevel(logging.INFO)

    ## high level api
    train_iter.reset()

    # create a module
    mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

    # fit the module
    mod.fit(train_iter,
            eval_data=val_iter,
            optimizer='sgd',
            optimizer_params={'learning_rate':0.1},
            eval_metric='acc',
            num_epoch=8)

    y = mod.predict(val_iter)
    assert y.shape == (4000, 26)

    score = mod.score(val_iter, ['acc'])
    print("Accuracy score is %f" % (score[0][1]))
    assert score[0][1] > 0.77, "Achieved accuracy (%f) is less than expected (0.77)" % score[0][1]

def get_mlp():
    outLabl = mx.sym.Variable('softmax_label')    
    data = mx.symbol.Variable('data')
    flat = mx.symbol.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = flat, name='fc1', num_hidden=20)
    act1 = mx.symbol.Activation(data = fc1, name='sig1', act_type="sigmoid")
    fc2  = mx.symbol.FullyConnected(data = act1, name='fc2', num_hidden=1)
    net  = mx.sym.LinearRegressionOutput(data=fc2, label=outLabl, name='linreg1')
    return net

# another one
def test_gaga_2():
    import logging
    logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # the usual data fetch
    data, yarr, features, fnames = getGagaData(maxrows=500, maxfeatures=500, gtype=None, stopwords='english', shuffle_=True)
    yarr = np.array(yarr).reshape(-1,1)

    # mxnet wrappers on data
    ntrain = int(data.shape[0]*0.7)
    ntrain_end = int(data.shape[0]*0.9)
    train_iter = mx.io.NDArrayIter(data[:ntrain], yarr[:ntrain], batch_size=32, shuffle=True)
    val_iter = mx.io.NDArrayIter(data[ntrain:ntrain_end], yarr[ntrain:ntrain_end], batch_size=16)
    test_iter = mx.io.NDArrayIter(data[ntrain_end:], yarr[ntrain_end:], batch_size=4)

    mlp = get_mlp()
    model = mx.mod.Module(context=mx.cpu(), symbol= mlp)
    model.fit(train_data=train_iter, eval_data=val_iter, optimizer_params={'learning_rate':0.05},
        batch_end_callback=mx.callback.Speedometer(1,100), eval_metric='mse', num_epoch=10)

    predictions = model.predict(test_iter)
    print(predictions)

# yet another gaga solver
def test_gaga_1():
    # the usual data fetch
    data, yarr, features, fnames = getGagaData(maxrows=500, maxfeatures=500, gtype=None, stopwords='english', shuffle_=True)
    yarr = np.array(yarr).reshape(-1,1)

    # mxnet wrappers on data
    ntrain = int(data.shape[0]*0.8)
    train_iter = mx.io.NDArrayIter(data[:ntrain], yarr[:ntrain], batch_size=32, shuffle=True)
    test_iter = mx.io.NDArrayIter(data[ntrain:], yarr[ntrain:], batch_size=32)

#    mnist = mx.test_utils.get_mnist()
#    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], 32, shuffle=True)
#    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], 32)


    x = mx.sym.Variable('data')
#    target = mx.sym.Variable('target')

    # x = mx.sym.FullyConnected(input, name='in1', num_hidden=500, no_bias=True)
    # x = mx.sym.FullyConnected(x, name='hid1', num_hidden=20, no_bias=True)
    # x = mx.sym.FullyConnected(x, name='hid2', num_hidden=1, no_bias=True)
    # x = mx.sym.Activation(x, name='sig1', act_type="sigmoid")
    # x = mx.sym.SoftmaxOutput(x, name="softmax", label=target)

    # The first fully-connected layer and the corresponding activation function
    x = mx.sym.FullyConnected(data=x, num_hidden=20)
    x = mx.sym.Activation(data=x, act_type="sigmoid")

    # The second fully-connected layer and the corresponding activation function
#    x = mx.sym.FullyConnected(data=x, num_hidden = 20)
#    x = mx.sym.Activation(data=x, act_type="sigmoid")

    # MNIST has 10 classes
    x = mx.sym.FullyConnected(data=x, num_hidden=1)
    # Softmax with cross entropy loss
    mlp  = mx.sym.SoftmaxOutput(data=x, name='softmax')

    mod = mx.mod.Module(symbol=mlp,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])

#    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)  # stuck on softmax_label
    mod.bind(data_shapes = [mx.io.DataDesc(name='data', shape=(32, 500))],
            label_shapes= [mx.io.DataDesc(name='softmax_label', shape=(32, 1))])
    mod.init_params(initializer=mx.init.Uniform(scale=.01))  #init random vals
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.01), ))  
    mod.init_params()
    mod.init_optimizer()

    metric = mx.metric.MSE()  

    for epoch in range(5):
        train_iter.reset()
        metric.reset()
        for batch in train_iter:
            mod.forward(batch, is_train=True)       # compute predictions
            mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
            mod.backward()                          # compute gradients
            mod.update()                            # update parameters
        print('Epoch %d, Training %s' % (epoch, metric.get()))

    # import logging
    # logging.getLogger().setLevel(logging.INFO)
    # mod.fit(train_data=train_iter, num_epoch=5)

    # import logging
    # logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout
    # # create a trainable module on compute context
    # mlp_model = mx.mod.Module(symbol=mlp, context=mx.cpu())
    # mlp_model.fit(train_iter,  # train data
    #             eval_data=test_iter,  # validation data
    #             optimizer='sgd',  # use SGD to train
    #             optimizer_params={'learning_rate':0.01},  # use fixed learning rate
    #             eval_metric='mse',  # report accuracy during training
    #             batch_end_callback = mx.callback.Speedometer(32, 100), # output progress for each 100 data batches
    #             num_epoch=3)  # train for at most 10 dataset passes

test_gaga_2()
