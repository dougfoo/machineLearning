import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
from mpmath import *
import logging as log

def setupTestData():
    df = pandas.read_csv('fakeGagaData.dat')
    return df

# generic solver takes in hypothesis function, cost func, training matrix, theta array, yarray
def grad_descent4(hFunc, cFunc, trainingMatrix, yArr):
    guesses = [0.1]*len(trainingMatrix[0])    # initial guess for all 
    step = 0.01          # init step
    step_limit = 0.001   # when to stop, when cost stops changing
    loop_limit = 50      # arbitrary max limits
    costChange = 1.0

    # TODO do i really need these 2 here... pass them in?
    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array
    
    log.warn('init guesses %s',str(guesses))
    log.warn('init func: %s, training size: %d' %(str(hFunc),trainingMatrix.shape[0]))
    log.debug('ts: %s / xs: %s',ts,xs)

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
    cost = 0.0+costF.subs(zip(ts,guesses))  
    log.warn('init cost: %f, costF %s',cost,str(costF)) # show first 80 char of cost evaluation

    i=0  
    while (abs(costChange) > step_limit and i<loop_limit):  # arbitrary limiter
        for j,theta in enumerate(ts):
            pd = evalPartialDeriv2(cFunc,theta,ts,xs,trainingMatrix,guesses,yArr)
            guesses[j] = guesses[j] - step * pd
        previousCost = cost
        cost = costF.subs(zip(ts,guesses))
        costChange = previousCost-cost
        log.warn('i=%d,costChange=%f,cost=%f, guesses=%s'%(i, costChange,cost,str(guesses)))
        i=i+1
    return guesses

# expnd to avg(sum(evaluated for testData)) 
def evalSumF2(f,xs,trainingMatrix,yArr):  # @TODO change testData to matrix
    assert (len(xs) == len(trainingMatrix[0]))
    assert (len(trainingMatrix) == len(yArr))
    n=0.0
    _f = f 
    for i,row in enumerate(trainingMatrix):
        for j,x in enumerate(xs):
            _f = _f.subs(x,row[j])
        n+= _f.subs(sp.symbols('y'),yArr[i])
        log.debug('_f: %s n: %s',_f, n)
        _f = f
        log.info('------ expand: y:(%d) %s to %s ', yArr[i],str(row),str(n))
    n *= (1.0/len(trainingMatrix))
    log.info('f (%d) %s - \n  ->expanded: %s '%(len(trainingMatrix[0]), str(f), str(n)))
    return n 

# generate deriv and sub all x's w/ training data and theta guess values
def evalPartialDeriv2(f,theta,ts,xs,trainingMatrix,guesses,yArr):
    pdcost = evalSumF2(sp.diff(f,theta),xs,trainingMatrix,yArr)
    pdcost = pdcost.subs(zip(ts,guesses))
    log.info ('    --> pdcost %f ;  %s  ;  %s: '%(pdcost,str(f),str(theta)))
    return pdcost

# test Logistic Regression v2
def testLR2():
    df = setupTestData()
    trainingMatrix = df.iloc[:,2:7].as_matrix()
    yArr = df.iloc[:,7:8].as_matrix()

    log.debug (df)
    log.debug (trainingMatrix)
    log.debug (yArr)

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    c,g,h,y = sp.symbols('c g h y')
    h = (sp.Matrix([ts])*sp.Matrix(xs))[0] # multipy ts's * xs's ( ts * xs.T )
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid
    c = -y*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    log.info ('g: %s',g)
    log.info ('c: %s',c)
    log.info ('tMatrix: %s',trainingMatrix)
    log.info ('yArr: %s',yArr)
    log.warn('columns: %s',df.head(0))
    grad_descent4(g,c,trainingMatrix,yArr)
    print 'done'

# store weights in theta[array] ?
# store xparams also in matrix
# store yresults in vector
# use sympy ?  maybe to start
# 

log.basicConfig(level=log.WARN)
log.info('start %s'%(log.getLogger().level))

#testLR()
testLR2()
