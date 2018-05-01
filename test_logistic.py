# content of test_sample.py
# test test
from myutils import *
import featureEngineering as fe
import pandas, inspect
from mpmath import *
import numpy as np
import logging as log

def inc(x):
    print (inspect.currentframe().f_code.co_name)
    return x + 1

def test_inc():
    print (inspect.currentframe().f_code.co_name)
    assert(4 == inc(3))

def test_lr_gaga_solver_1():
    print (inspect.currentframe().f_code.co_name)
    yArr = [1,0]
    trainingMatrix = np.array([[4,0],[0,5]])  #dummy data 1's, word1, word2 -- first row gaga, 2nd non

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    f = (sp.Matrix(ts).T * sp.Matrix(xs) ) [0]
    g = 1 / (1+mp.e**-f)   # wrap in sigmoid
    y = sp.symbols('y')
    cFunc = (0-y)*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData

    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.01,step_limit=0.00001,loop_limit=50)    
    assert (round(gs[0],2) == 0.33)
    assert (round(gs[1],2) == -0.34)

def test_lr_gaga_solver_2():
    print (inspect.currentframe().f_code.co_name)
    yArr = [1,1,0,0]
    trainingMatrix = np.array([[12,5,1,2],[10,5,3,1],[0,1,8,2],[2,1,7,7]])  #dummy data 1's, word1, word2 -- first row gaga, 2nd non

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    f = (sp.Matrix(ts).T * sp.Matrix(xs) ) [0]
    g = 1 / (1+mp.e**-f)   # wrap in sigmoid
    y = sp.symbols('y')
    cFunc = (0-y)*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData

    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.01,step_limit=0.00001,loop_limit=10,batchSize=2)    
    assert (round(gs[0],2) == 0.15)
    assert (round(gs[1],2) == 0.05)
    assert (round(gs[2],2) == -0.12)
    assert (round(gs[3],2) == -0.06)

#    assert(round(gs[0]) == 11)
#    assert(round(gs[1],1) == 1.5)    

log.getLogger().setLevel(log.ERROR)

test_lr_gaga_solver_1()
test_lr_gaga_solver_2()




