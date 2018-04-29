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
import logging as log

def getTestData():
    df = pandas.read_csv('fakeGagaData.dat')
    return df

# test with mike eng's dataset
def testLRGaga():
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

def testLRGaga2(kFeatures=50,bs=4,ts=10):
    print 'test gaga solver'
    trainingMatrix,yArr,labels,fnames = fe.getGagaData(maxrows=ts,stopwords='english')
    t1 = np.array(trainingMatrix)

    # SelectKBest 
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X, y = t1, yArr
    log.warn('Orig size: %s'%(str(X.shape)))
    model = SelectKBest(chi2, k=kFeatures)
    X_new = model.fit_transform(X, y)   # need to keep labels
    log.warn('KBest %d applied %s'%(kFeatures, str(X_new.shape)))
    df = pandas.DataFrame(X_new)
    picklist = model.get_support(True)
    pickwords = [labels[p] for p in picklist]

    # reduce to K features
    df = pandas.DataFrame(trainingMatrix, columns=labels)
    df = df[pickwords]
    print(df.shape)
    trainingMatrix = df.as_matrix()
    labels = pickwords
 
    m1,c1 = fe.countWords2(trainingMatrix, labels, fnames)
    log.warn('reduced matrix: %s'%str(m1))

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array
    xt = sp.Matrix(ts).T * sp.Matrix(xs)
    f = xt[0]
    g = 1 / (1+mp.e**-f)   # wrap in sigmoid
    y = sp.symbols('y')
    cFunc = -y*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData

    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    X = np.asmatrix(trainingMatrix)
    sol = (X.T.dot(X)).I.dot(X.T).dot(yArr)
    sol = np.asarray(sol)[0]
    log.warn ('target sol (X.T * X)^-1 * X.T*Y:     %s'% str(fe.gf(sol)))

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.001,step_limit=0.00001,loop_limit=500, batchSize=bs)    
    log.warn('scaled A: %f'%(gs[0]))
    log.warn('scaled B: %f'%(gs[1]))

    log.warn ('target sol (X.T * X)^-1 * X.T*Y:     %s'% str(fe.gf(sol)))

#log.basicConfig(level=log.WARN)
#log.info('start %s'%(log.getLogger().level))

#testLRGaga()
