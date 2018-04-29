import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
from mpmath import *
import logging as log

def getTestData():
    df = pandas.read_csv('fakeGagaData.dat')
    return df

# test Logistic Regression v2
def testLR2():
    df = getTestData()
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

    log.warn ('g: %s',g)
    log.warn ('c: %s',c)
    log.warn ('tMatrix: %s',trainingMatrix)
    log.warn ('yArr: %s',yArr)
    log.warn('columns: %s',df.head(0))
    grad_descent4(g,c,trainingMatrix,yArr)
    print 'done'

# test with mike eng's dataset
def testGaga():
    trainingMatrix,yArr,labels,fnames = getGagaData(maxrows=300,maxfeatures=20)

    log.debug (trainingMatrix)
    log.debug (yArr)

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    c,g,h,y = sp.symbols('c g h y')
    h = (sp.Matrix([ts])*sp.Matrix(xs))[0] # multipy ts's * xs's ( ts * xs.T )
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid 
    c = -y*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    log.warn ('g: %s',g)
    log.warn ('c: %s',c)
    log.warn ('tMatrix: %s',trainingMatrix)
    log.warn ('yArr: %s',yArr)
    log.warn('columns: %s',labels)
    grad_descent4(g,c,trainingMatrix,yArr)
    print 'done'

log.basicConfig(level=log.WARN)
log.info('start %s'%(log.getLogger().level))
#testLR2()
testGaga()
