from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from myutils import *
from sklearn.utils import shuffle


def test_basics():
    x = torch.empty(5, 3)
    print(x)

    x = x.new_ones(5, 3)
    print(x)
    """ 
    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
    print(x)

    x = torch.randn_like(x, dtype=torch.float)    # override dtype!
    print(x)    """

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

# re-done base case logistic regression using bits of pytorch
def test_gaga_lr():
    # input specific setup
    data, yarr, features, fnames = getGagaData(maxrows=200, maxfeatures=2000, gtype=None, stopwords='english')
    xMatrix = shuffle(data, random_state=0)
    yArr = shuffle(np.array(yarr).reshape(-1,1), random_state=0)
    partition = int(.70*len(yArr))
    trainingX = xMatrix[:partition]
    trainingY = yArr[:partition]
    testX = xMatrix[partition:]
    testY = yArr[partition:]

    # wrap in torch functions
    dtype = torch.float
    x = torch.tensor(trainingX, dtype=dtype)
    y = torch.tensor(trainingY, dtype=dtype)  
    testx = torch.tensor(testX, dtype=dtype)
    testy = torch.tensor(testY, dtype=dtype)  

    # Randomly initialize weights
    torch.manual_seed(0)
    w1 = torch.randn(len(features),1, dtype=dtype)  # (n,1) shape

    learning_rate = 0.1
    for t in range(25000):
        h = x.mm(w1)               # each feature * weight
        y_pred = h.sigmoid()
        loss = (y_pred - y).pow(2).sum().item()  
        if (t % 5000 == 0):
            log.warn('loop %d, %.8f'%(t, loss))

        grad_w1 = x.t().mm(y_pred - y)
        w1 -= learning_rate * grad_w1

    print('training complete ', loss)
    print('test validation phase')

    h = testx.mm(w1)               # matrixMult or dot prod == same?
    y_pred = h.sigmoid().round()

    log.debug('ytest', pandas.DataFrame(testy.numpy()).head())
    log.debug('ypred', pandas.DataFrame(y_pred.numpy()).head())

    # Compute and print loss after rounding to 0/1's
    testDiffs = (y_pred - testy)
    p = pandas.DataFrame(testDiffs.numpy())
    log.debug('diffs', p.head())
    tests = len(p)
    correct = len(p[(p[0] == 0)])
    print('total correct/tests', correct, tests)
    print('correct % =', round((correct/tests)*100, 2))

# torch neural net manual
def test_gaga_nn():
    dtype = torch.float
    device = torch.device("cpu")
    N = 200 # training examples
    F = 2000 # features
    H = 400 # hidden nodes
    D_in, D_out = F, 2  # 100 hidden nodes, 2 output nodes

    data, yarr, features, fnames = getGagaData(maxrows=N, maxfeatures=F, gtype=None, stopwords='english')
    xMatrix = shuffle(data, random_state=0)
    yArr = shuffle(yarr, random_state=0)

    partition = int(.70*len(yArr))
    trainingX = xMatrix[:partition]
    trainingY = yArr[:partition]
    testX = xMatrix[partition:]
    testY = yArr[partition:]

    # Create random input and output data
    x = torch.tensor(trainingX, dtype=dtype)
    y = torch.tensor(pd.get_dummies(trainingY).values, dtype=dtype)  # onehot y's
    testx = torch.tensor(testX, dtype=dtype)
    testy = torch.tensor(pd.get_dummies(testY).values, dtype=dtype)  # onehot y's

    # Randomly initialize weights, repeatable w/ seed
    torch.manual_seed(0)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, dtype=dtype)

    #gradient descent
    learning_rate = 0.0001
    for t in range(25000):
        # Forward pass: compute predicted y
        h = x.mm(w1)               # matrixMult or dot prod == same?
        h_relu = h.clamp(min=0)    # min/max = clamp function, builds relu
        y_pred = h_relu.mm(w2)     

        loss = (y_pred - y).pow(2).sum().item()  # item unwraps
        if (t % 1000 == 0):
            print(t, loss)
            if (loss < 0.0001):
                break

        # Manual backprop routines
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred) 
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0   # 0 out negatives, like relu ???
        grad_w1 = x.t().mm(grad_h)

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    print('training complete ',loss)
    print('test validation phase')

    h = testx.mm(w1)               # matrixMult or dot prod == same?
    h_relu = h.clamp(min=0)    # min/max = clamp function
    y_pred = h_relu.mm(w2)
    y_pred2 = pd.get_dummies(y_pred.argmax(dim=1))   # to 0,1 -> 2 col arr
    y_pred2 = torch.tensor(y_pred2.values, dtype=dtype)

    print('ytest', pandas.DataFrame(testy.numpy()).head())
    print('ypred', pandas.DataFrame(y_pred.numpy()).head())
    print('ypred2', pandas.DataFrame(y_pred2.numpy()).head())

    # Compute and print loss after rounding to 0/1's
    testDiffs = (y_pred2 - testy)
    p = pandas.DataFrame(testDiffs.numpy())
    print('diffs', p.head())
    tests = len(p)
    correct = len(p[(p[0]==0) & (p[1]==0)])
    print('total correct/tests',correct,tests)
    print('correct % =', round((correct/tests)*100,2))
    print('hidden nodes %d'%H)

def test_gaga_nn_auto():
    dtype = torch.float
    device = torch.device("cpu")
    N = 200  # training examples
    F = 2000  # features
    D_in, H, D_out = F, 500, 2  # 100 hidden nodes, 2 output nodes

    data, yarr, features, fnames = getGagaData(maxrows=N, maxfeatures=F, gtype=None, stopwords='english')
    xMatrix = shuffle(data, random_state=0)
    yArr = shuffle(yarr, random_state=0)

    partition = int(.70*len(yArr))
    trainingX = xMatrix[:partition]
    trainingY = yArr[:partition]
    testX = xMatrix[partition:]
    testY = yArr[partition:]

    # Create random input and output data
    x = torch.tensor(trainingX, dtype=dtype)
    y = torch.tensor(pd.get_dummies(trainingY).values, dtype=dtype)  # onehot y's
    testx = torch.tensor(testX, dtype=dtype)
    testy = torch.tensor(pd.get_dummies(testY).values,
                         dtype=dtype)  # onehot y's

    # Randomly initialize weights, repeatable w/ seed
    torch.manual_seed(0)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, dtype=dtype, requires_grad=True)

    #gradient descent using autograd
    learning_rate = 0.000001
    for t in range(25000):
        # Forward pass: compute predicted y
        h = x.mm(w1)               # matrixMult or dot prod == same?
        h_relu = h.clamp(min=0)    # min/max = clamp function, builds relu
        y_pred = h_relu.mm(w2)

        loss = (y_pred - y).pow(2).sum().item()  # item unwraps
        if (t % 1000 == 0):
            print(t, loss)
            if (loss < 0.0001):
                break

        if grad_w1 is not None:
            grad_w1.grad.zero_()
        if grad_w2 is not None:
            grad_w2.grad.zero_()
        loss.backward()

        grad_w1 = w1.grad()
        grad_w2 = w2.grad()

        # Manual backprop routines
        '''
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0   # 0 out negatives, like relu ???
        grad_w1 = x.t().mm(grad_h)
'''
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    print('training complete ', loss)
    print('test validation phase')

    h = testx.mm(w1)               # matrixMult or dot prod == same?
    h_relu = h.clamp(min=0)    # min/max = clamp function
    y_pred = h_relu.mm(w2)
    y_pred2 = pd.get_dummies(y_pred.argmax(dim=1))   # to 0,1 -> 2 col arr
    y_pred2 = torch.tensor(y_pred2.values, dtype=dtype)

    print('ytest', pandas.DataFrame(testy.numpy()).head())
    print('ypred', pandas.DataFrame(y_pred.numpy()).head())
    print('ypred2', pandas.DataFrame(y_pred2.numpy()).head())

    # Compute and print loss after rounding to 0/1's
    testDiffs = (y_pred2 - testy)
    p = pandas.DataFrame(testDiffs.numpy())
    print('diffs', p.head())
    tests = len(p)
    correct = len(p[(p[0] == 0) & (p[1] == 0)])
    print('total correct/tests', correct, tests)
    print('correct % =', round((correct/tests)*100, 2))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
print ('--------------')
test_gaga_nn()
#test_gaga_lr()

'''
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print('a',a)
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print('b=a*a',b)
print('b',b.grad_fn)

print(z, out)
out.backward()
print(z, out)

print(x.grad)
'''
