import requests, pandas, io, numpy, argparse
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

    h = ts[0]*xs[0] + ts[1]*xs[1] + ts[2]*xs[2] + ts[3]*xs[3] + ts[4]*xs[4] + ts[5]*xs[5] 
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid
    c = y*-math.log(x) + (1-y)*-math.log(1-x)  # cost func of single sample
    
    print ('init guesses',guesses)
    print ('init func: %s, test size: %d' %(str(f),testData.shape[0]))
    
    costF = evalSumF(c,ts,xs,testData)  # cost fun evaluted for testData
    print('init costF',str(costF)[:80]) # show first 80 char of cost evaluation
    costEval = costF.subs(ts,guesses)  # cost evaluted for all terms guess
    print('init cost',costEval)

    i=0  
    while (abs(costChange) > step_limit and i<loop_limit):  # arbitrary limiter
        j=0
        for theta in ts:
            pdb = evalPartialDeriv(e,theta,xs[j],testData,guesses[j])
            guesses[j] = guesses[j] - stepA * pda
        previousCost = costEval
        costEval = costF.subs(ts, guesses)
        costChange = previousCost-costEval
        print ('i=%d,cost=%d'%(i, int(costEval), guessA, guessB), guesses)
        i=i+1
    return guesses

def toMatrix(df):
    x,y = sp.symbols('x y')

    X = df.iloc[:,1:6]
    Y = df['gaga']

    X = numpy.asmatrix(X.as_matrix())
    Y = Y.as_matrix()
    print X,Y,x
    return X,Y

def toSymbols(df):
    g,h,y = sp.symbols('g h y')
    t = sp.symbols('t0 t1 t2 t3 t4 t5')
    x = sp.symbols('x0 x1 x2 x3 x4 x5')

    h = t[0]*x[0] + t[1]*x[1] + t[2]*x[2] + t[3]*x[3] + t[4]*x[4] + t[5]*x[5] 
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid

    print x
    print t
    print h
    print g


# test runners
def testLR():
    df = setupTestData()
    print df
    toMatrix(df)
    toSymbols(df)
    print 'done'

# store weights in theta[array] ?
# store xparams also in matrix
# store yresults in vector
# use sympy ?  maybe to start
# 


testLR()
