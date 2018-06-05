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

def test_nn_basic():
    dtype = torch.float
    device = torch.device("cpu")
    # dtype = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    print (type(x))

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, dtype=dtype)

    learning_rate = 1e-6
    for t in range(500):
        # Forward pass: compute predicted y
        h = x.mm(w1)               # matrixMult or dot prod == same?
        h_relu = h.clamp(min=0)    # min/max = clamp function
        y_pred = h_relu.mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()  # item?
        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2


def test_nn_gaga():
    dtype = torch.float
    device = torch.device("cpu")
    # dtype = torch.device("cuda:0") # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N = 200 # examples
    F = 2000 # features
    D_in, H, D_out = F, 100, 2

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

    # Randomly initialize weights
    torch.manual_seed(0)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, dtype=dtype)

    learning_rate = 1e-6
    for t in range(5000):
        # Forward pass: compute predicted y
        h = x.mm(w1)               # matrixMult or dot prod == same?
        h_relu = h.clamp(min=0)    # min/max = clamp function
        y_pred = h_relu.mm(w2)
        #y_pred = h.mm(w2).sigmoid()

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()  # item?
        if (t % 500 == 0):
            print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        #grad_w2 = h.t().mm(grad_y_pred)
        #grad_h = grad_y_pred.mm(w2.t()).clone()
        grad_h[h < 0] = 0   # 0's them out
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

    print('training complete ',loss)
    print('test validation phase')

    h = testx.mm(w1)               # matrixMult or dot prod == same?
    h_relu = h.clamp(min=0)    # min/max = clamp function
    y_pred = h_relu.mm(w2)
    y_pred2 = pd.get_dummies(y_pred.argmax(dim=1))   # to 0,1 -> 2 col arr
    y_pred2 = torch.tensor(y_pred2.values, dtype=dtype)

    # y_pred = h.mm(w2)
    # y_pred_sig = y_pred.sigmoid()
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
#test_nn_basic()
#test_nn_gaga()

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
