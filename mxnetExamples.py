import mxnet as mx
import numpy as np
from myutils import *
from nnutils import *


# completely failed to make it work
# in mxnet, i blame the documentation

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

def test_gluon():
    import mxnet as mx
    from mxnet import gluon
    from mxnet.gluon import nn
    from mxnet import autograd as ag

    # Fixing the random seed
    mx.random.seed(0)
    mnist = mx.test_utils.get_mnist()

    batch_size = 100
#    train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
#   val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    # define network
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(128, activation='relu'))
        net.add(nn.Dense(64, activation='relu'))
        net.add(nn.Dense(10))

    gpus = mx.test_utils.list_gpus()
    ctx =  [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.02})

    # training loop
    # Use Accuracy as the evaluation metric.
    epoch = 10
    metric = mx.metric.Accuracy() # or MSE()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(epoch):
        # Reset the train data iterator.
        train_data.reset()
        # Loop over the train data iterator.
        for batch in train_data:
            # Splits train data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            # Inside training scope
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    # Computes softmax cross entropy loss.
                    loss = softmax_cross_entropy_loss(z, y)
                    # Backpropagate the error for one iteration.
                    loss.backward()
                    outputs.append(z)
            # Updates internal evaluation
            metric.update(label, outputs)
            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            trainer.step(batch.data[0].shape[0])
        # Gets the evaluation result.
        name, acc = metric.get()
        # Reset evaluation result to initial state.
        metric.reset()
        print('training mse at epoch %d: %s=%f'%(i, name, acc))

def test_gluon_gaga():
    import mxnet as mx
    from mxnet import gluon
    from mxnet.gluon import nn
    from mxnet import autograd as ag

    # Fixing the random seed
    mx.random.seed(0)
    mnist = mx.test_utils.get_mnist()

    batch_size = 32
#    train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
#   val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    data, yarr, features, fnames = getGagaData(maxrows=500, maxfeatures=500, gtype=None, stopwords='english', shuffle_=True)
    yarr = np.array(yarr).reshape(-1,1)
    ntrain = int(data.shape[0]*0.8)
    train_data = mx.io.NDArrayIter(data[:ntrain], yarr[:ntrain], batch_size=batch_size, shuffle=True)
    val_data = mx.io.NDArrayIter(data[ntrain:], yarr[ntrain:], batch_size=batch_size)

    # define network
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(500, activation='relu'))
        net.add(nn.Dense(20, activation='relu'))
        net.add(nn.Dense(2))

    gpus = mx.test_utils.list_gpus()
    ctx =  [mx.gpu()] if gpus else [mx.cpu(0), mx.cpu(1)]
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)  #relu=xavier
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    # training loop
    # Use Accuracy as the evaluation metric.
    epoch = 1500
    metric = mx.metric.MSE() # or MSE()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()  # log loss
    for i in range(epoch):
        # Reset the train data iterator.
        train_data.reset()
        # Loop over the train data iterator.
        for batch in train_data:
            # Splits train data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            # Inside training scope
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    # Computes softmax cross entropy loss.
                    loss = softmax_cross_entropy_loss(z, y)
                    # Backpropagate the error for one iteration.
                    loss.backward()
                    outputs.append(z)
            # Updates internal evaluation
            metric.update(label, outputs)
            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            trainer.step(batch.data[0].shape[0])
        # Gets the evaluation result.
        name, acc = metric.get()
        # Reset evaluation result to initial state.
        metric.reset()
        if (i % 20 ==0):
            print('training mse at epoch %d: %s=%f'%(i, name, acc))

    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    # Reset the validation data iterator.
    val_data.reset()
    # Loop over the validation data iterator.
    for batch in val_data:
        # Splits validation data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits validation label into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x))
        # Updates internal evaluation
        metric.update(label, outputs)
    print('validation acc: %s=%f'%metric.get())
#    assert metric.get()[1] > 0.94

def test_gluon_logr_gaga():
    import mxnet as mx
    from mxnet import gluon
    from mxnet.gluon import nn
    from mxnet import autograd as ag
    from mxnet import nd, autograd
    from mxnet import gluon
    import numpy as np

    batch_size = 10
    num_outputs = 1
    model_ctx = mx.cpu()

    def evaluate_accuracy(data, y, net):
        acc = mx.metric.Accuracy()
        for data, label in zip(data,y):
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    data, yarr, features, fnames = getGagaData(maxrows=500, maxfeatures=500, gtype=None, stopwords='english', shuffle_=True)
    data = nd.array(data)
    yarr = nd.array(np.array(yarr).reshape(-1,1))
    ntrain = int(data.shape[0]*0.8)
    train_data = data[:ntrain]
    test_data = data[ntrain:]
    train_y = yarr[:ntrain]
    test_y = yarr[ntrain:]

    net = gluon.nn.Dense(num_outputs)
    net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
    evaluate_accuracy(test_data, test_y, net)
    epochs = 10

    for e in range(epochs):
        cumulative_loss = 0
        for data,label in enumerate(zip(train_data,train_y)):
            with ag.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            cumulative_loss += nd.sum(loss).asscalar()

        test_accuracy = evaluate_accuracy(test_data, net)
        train_accuracy = evaluate_accuracy(train_data, net)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, cumulative_loss/100, train_accuracy, test_accuracy))


test_gluon_logr_gaga()


