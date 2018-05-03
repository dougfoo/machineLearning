import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import featureEngineering as fe
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
from mpmath import *
from gdsolvers import *
import logging as log
import inspect


def test_grad_descent5_logr_vs_ref():
    print (inspect.currentframe().f_code.co_name)
    X = np.asarray([
        [0.50],[0.75],[1.00],[1.25],[1.50],[1.75],[1.75],
        [2.00],[2.25],[2.50],[2.75],[3.00],[3.25],[3.50],
        [4.00],[4.25],[4.50],[4.75],[5.00],[5.50]])
    ones = np.ones(X.shape)  
    X = np.hstack([ones, X])  # makes it [[1,.5][1,.75]...]
    Y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1])
    Y2 = Y.reshape([-1, 1])  # reshape Y so it's column vector so matrix multiplication is easier

    gs = grad_descent5(lambda y,x: sigmoid(x)-y,log_loss,X,Y,step=0.5,step_limit=0.0000001,loop_limit=5000, batchSize=30)    
    log.warn('final: %s'%gs)
    gs2 = gradient_descent_logr(X, Y2, 5000, 0.5)
    print ('grad_logr',gs2)

    gs3 = sklearn_comp(X,Y)
    print ('scikit logreg',gs3)

    assert(round(gs[0],2) == round(gs2[0],2))
    assert(round(gs[1],2) == round(gs2[1],2))

if __name__ == "__main__":
    log.getLogger().setLevel(log.WARN)

    test_grad_descent5_logr_vs_ref()


