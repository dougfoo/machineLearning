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

# generic solver for logistic regression take in array of theta and y
def grad_descent3(f, cf, testData=setupData(), pltAx=False, batchSize=None, t='grad_desc'):
    guessA = guessB = 1.0   #initial guess y=1x+1
    step  = 0.001
    stepA = 0.00000005   #dif step for diff A,B ?
    stepB = 0.25         #maybe normalize data first
    step_limit = 1.0    # when to stop, when cost change < step_limit
    loop_limit = 1000    # arbitrary max limits
    costChange = step_limit+1

    A,B,x,y = sp.symbols('A B x y')
    e = cf
    print ('init guess A: %f, B: %f'%(guessA,guessB))
    print ('init func: %s, test size: %d' %(str(f),testData.shape[0]))
    costF = evalSumF(e,x,y,testData)  # cost fun evaluted for testData
    print('init costF',str(costF)[:80])  # oddly this line crashes on yoga tablet
    costEval = costF.subs(A,guessA).subs(B,guessB)  # cost evaluted for A B guess
    print('init cost',costEval)

    # add optional plot - scatter of testData
    if (pltAx):
        pltAx=plotScatter(testData,xLabel='head_size',yLabel='brain_weight')
        pltAx.set_title(t)
        max = testData['head_size'].max()
        min = testData['head_size'].min()
        print(pltAx, max,min)

    i=j=l=0
    if (batchSize == None):
        batchSize = len(testData)  #@todo can i set this in func param

    # outer loop std grad descent solver loop
    while (abs(costChange) > step_limit and l<loop_limit):  # arbitrary limiter
        i=j=k=0
        testData = shuffle(testData)
        k = j+batchSize if j+batchSize<len(testData) else len(testData)
        dataBatch = testData[j:k]
#        print('shuffled loop - batch size: %d, j: %d, k: %d, %d'%(batchSize,j,k, len(testData)/batchSize))

        # inner batch of size batchSize - test in batches and redo again
        while (i < len(testData)/batchSize):
            pda = evalPartialDeriv(e,x,y,dataBatch,A,guessA,B,guessB)
            pdb = evalPartialDeriv(e,x,y,dataBatch,B,guessB,A,guessA)
            guessA = guessA - stepA * pda
            guessB = guessB - stepB * pdb
            previousCost = costEval
            costEval = costF.subs(A,guessA).subs(B,guessB)
            costChange = previousCost-costEval
            print ('t=%s,l=%d,i=%d,cost=%d,A=%f,B=%f,bs=%d'%(t, l, i, int(costEval), guessA, guessB, batchSize))
            # add optional plot of current regression line
            if (pltAx):
                plotLine(pltAx,guessA,guessB,min,max)
            j = k
            k = j+batchSize if j+batchSize<len(testData) else len(testData)
            dataBatch = testData[j:k]
            i += 1
            l += 1
    return guessA,guessB

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
