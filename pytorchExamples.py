from __future__ import print_function
import torch
import numpy as np

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


x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)


