import tensorflow as tf
import torch, mxnet, numpy

# multi model wrapping tensorflow, pytorch and mxnet libraries
# to compare diff implementations of standard NN models
class MultiModel:
    def __init__(self, name):
        self.name=name
        print('init')

    def forward(self):
        print('forward propagate')

    def backward(self):
        print('back prop')


# main
mm = MultiModel('myModel')
mm.forward()
mm.backward()

