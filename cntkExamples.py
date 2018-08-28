from __future__ import print_function
import numpy as np
import cntk
from myutils import *
from nnutils import *
from sklearn.utils import shuffle
from torch.autograd import Variable
import torchvision.models as models

def test_cnn():
    from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
    import requests
    import os

    def download(url, filename):
        """ utility function to download a file """
        response = requests.get(url, stream=True)
        with open(filename, "wb") as handle:
            for data in response.iter_content():
                handle.write(data)

    locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

    data = {
    'train': { 'file': 'atis.train.ctf', 'location': 0 },
    'test': { 'file': 'atis.test.ctf', 'location': 0 },
    'query': { 'file': 'query.wl', 'location': 1 },
    'slots': { 'file': 'slots.wl', 'location': 1 },
    'intent': { 'file': 'intent.wl', 'location': 1 }
    }

    for item in data.values():
        location = locations[item['location']]
        path = os.path.join('..', location, item['file'])
        if os.path.exists(path):
            print("Reusing locally cached:", item['file'])
            # Update path
            item['file'] = path
        elif os.path.exists(item['file']):
            print("Reusing locally cached:", item['file'])
        else:
            print("Starting download:", item['file'])
            url = "https://github.com/Microsoft/CNTK/blob/release/2.5.1/%s/%s?raw=true"%(location, item['file'])
            download(url, item['file'])
            print("Download completed")

def test_basics():
    x = torch.empty(5, 3)
    print(x)

    x = x.new_ones(5, 3)
    print(x)

    y = torch.rand(5, 3)
    print (y)
    print(x + y)
    print(x.add(y))

    res = torch.empty(5,3)
    res.new_ones(5,3)
    print('res a',res)
    torch.add(x,y,out=res)
    print('res b',res)

    a = np.ones(5)
    b = torch.from_numpy(a)
    print (type(b), b)
    print (type(a), a)
    c = b.numpy()
    print (type(c), c)

    print ('cuda?', torch.cuda.is_available())


def test_cntk_gaga():
    net = Net()
    print(net)

    params = list(net.parameters())  
    print(len(params))   # list of major params matrices
    print(params[0].size())  # conv1's .weight

    input = torch.randn(1, 1, 32, 32)  # initial dummy input
    output = net(input)
    target = torch.arange(1, 11)  # a dummy target, for example
    target = target.view(1, -1)  # what is this ??
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print('loss tree',loss)
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

    # backprop
    loss.backward()
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    # sample backprop otpmizer
    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=0.01) # learning rate 0.01 / for each parameter matrix

    # in your training loop:
    for epoch in range(500):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
        if (epoch % 100 == 0):
            print('Epoch %s MSE %s' % (epoch, loss))

    print('conv1.bias.grad after optimizer')
    print(net.conv1.bias.grad)
    print ('--training done---')

    print('--test start---')
    test_input = torch.randn(1, 1, 32, 32)  # initial dummy input
    print ('input',test_input)
    print ('output/prediction',net.forward(test_input))   # returns activation

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-t', help='batch size', type=int)
args = vars(parser.parse_args())

_t = args['t']

test_cntk_gaga(_t)
#test_pytorch_nn()
#print('--------------******-----------')
#test_gaga_nn_auto()
#test_gaga_lr()


