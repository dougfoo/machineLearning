import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
from mpmath import *

def setupTestData():
    df = pandas.read_csv('fakeGagaData.dat')
    return df

# solver for logistic regression
# psuedocode for now
def grad_descent3(testData):
    guesses = [1.0]*6    # initial guess for all 
    step = 0.0001        # init step
    step_limit = 0.0001  # when to stop, when cost stops changing
    loop_limit = 5       # arbitrary max limits
    costChange = 1.0

    c,g,h,y,x = sp.symbols('c g h y x')
    ts = sp.symbols('t0 t1 t2 t3 t4 t5')  #theta weight/parameter array
    xs = sp.symbols('x0 x1 x2 x3 x4 x5')  #feature array

#    h = ts[0]*xs[0] + ts[1]*xs[1] + ts[2]*xs[2] + ts[3]*xs[3] + ts[4]*xs[4] + ts[5]*xs[5] 
    h = ts[0]*xs[0] + ts[1]*xs[1]   # start w/ just 2 terms 
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid
    c = y*-sp.log(g) + (1-y)*-sp.log(1-g)  # cost func of single sample
    
    print ('g: ',g)
    print ('c: ',c)

    print ('init guesses',guesses)
    print ('init func: %s, test size: %d' %(str(g),testData.shape[0]))
    
    costF = evalSumF(c,xs,testData)  # cost fun evaluted for testData
    print('init costF',str(costF)[:80]) # show first 80 char of cost evaluation
    cost = 0.0+costF.subs(ts[0],guesses[0]).subs(ts[1],guesses[1])  
    print('init cost',cost,type(cost))

    i=0  
    while (abs(costChange) > step_limit and i<loop_limit):  # arbitrary limiter
        j=0
        for theta in ts:
            pd = evalPartialDeriv(c,theta,ts,xs,testData,guesses)
            guesses[j] = guesses[j] - step * pd
        previousCost = cost
        _costF=costF
        for t,g in zip(ts,guesses):
            _costF = _costF.subs(t, g) 
        cost = _costF
        costChange = previousCost-cost
        print ('i=%d,cost=%f'%(i, cost), guesses)
        i=i+1
    return guesses

# expnd to avg(sum(evaluated for testData))
def evalSumF(f,xs,testData):
    n=0.0
    for _,d in testData.iterrows():  # global test data
        n += f.subs(xs[0],d.animal).subs(xs[1],d.vegetable).subs(sp.symbols('y'),d.gaga)
    n *= (1.0/len(testData))
    print('f %s expanded (%d) n: %s '%(str(f),len(testData),str(n)))
    return n 

# generate deriv and sub all x's w/ training data and theta guess values
def evalPartialDeriv(f,theta,ts,xs,testData,guesses):
    pdcost = evalSumF(sp.diff(f,theta),xs,testData)
    for t,g in zip(ts,guesses):
        pdcost = pdcost.subs(t,g)
    print ('pdcost f/t: ',pdcost,str(f),theta)
    return pdcost

def toMatrixTest(df):
    x,y = sp.symbols('x y')
    X = df.iloc[:,1:6]
    Y = df['gaga']
    print('pre matrix X,Y',X,Y)
    X = numpy.asmatrix(X.as_matrix())
    Y = Y.as_matrix()
    print('post as matrix X,Y',X,Y)
    return X,Y

def toSymbolsTest(df):
    g,h,y = sp.symbols('g h y')
    t = sp.symbols('t0 t1 t2 t3 t4 t5')
    x = sp.symbols('x0 x1 x2 x3 x4 x5')
#    h = t[0]*x[0] + t[1]*x[1] + t[2]*x[2] + t[3]*x[3] + t[4]*x[4] + t[5]*x[5] 
    h = t[0]*x[0] + t[1]*x[1]
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid

    print('x:',x)
    print('t:',t)
    print('h:',h)
    print('g:',g)

    guesses = [1,1]
    evalPartialDeriv(g, t[0], t[0:2], x[0:2], df, guesses)

# test runners
def testLR():
    df = setupTestData()
    print df
    toMatrixTest(df)
    toSymbolsTest(df)
    grad_descent3(df)
    print 'done'

# store weights in theta[array] ?
# store xparams also in matrix
# store yresults in vector
# use sympy ?  maybe to start
# 


testLR()
