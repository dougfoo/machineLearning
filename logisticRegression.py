import requests, pandas, io, numpy, argparse, math
import numpy as np
import featureEngineering as fe
from myutils import *
from mpmath import *
from gdsolvers import *
import logging as log
import inspect
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import shuffle

# take in x-nparray matrix, and return reduced matrix and labels
def reduceFeatures(X, Y, labels, kFeatures):
    # SelectKBest 
    log.warn('feature reduce KBest: %s'%(str(X.shape)))
    model = SelectKBest(chi2, k=kFeatures)
    X_new = model.fit_transform(X, Y)   # need to keep labels
    log.warn('KBest %d applied %s'%(kFeatures, str(X_new.shape)))
    df = pandas.DataFrame(X_new)
    indexes = model.get_support(True)
    words = [labels[p] for p in indexes]

    # reduce to K features
    df = pandas.DataFrame(X, columns=labels)
    df = df[words]
    X_new = df.as_matrix()

    return X,words

# test with mike eng's dataset
def test_lady_gaga_gd5_1(kFeatures=50,maxRows=100,loops=100):
    print (inspect.currentframe().f_code.co_name)
    xMatrix,yArr,labels,fnames = fe.getGagaData(maxrows=maxRows,stopwords='english')

    xMatrix = shuffle(xMatrix, random_state=0)   
    yArr = shuffle(yArr, random_state=0) 

    partition = int(.70*len(yArr))

    trainingMatrix = xMatrix[:partition]
    trainingY = yArr[:partition]
    testMatrix = xMatrix[partition:]
    testY = yArr[partition:]

    X = np.array(trainingMatrix)
    Y = trainingY

    X,rlabels = reduceFeatures(X, Y, labels, kFeatures)
 
    m1,c1 = fe.countWords2(X, labels, fnames)
    log.warn('reduced matrix: %s'%str(m1)) 

    gs = grad_descent5(lambda y,x: sigmoid(x)-y,sigmoidCost,X,Y,step=0.1,step_limit=0.000000001,loop_limit=loops)
    log.warn('guesses: %s', gf(gs))
    
    testRes = np.dot(testMatrix, gs)
    testResRound = [round(sigmoid(x),0) for x in testRes]
    testDiffs = np.array(testResRound) - np.array(testY)
    print ('raw results',gf(testRes))
    print ('sig results',gf([sigmoid(x) for x in testRes]))
    print ('0|1 results',[round(sigmoid(x)) for x in testRes])
    print (testDiffs)
    print ('total errors:', sum([abs(x) for x in testDiffs]), 'out of',len(testY))
    
#def foo():    
    # sklearn validation
    X = np.asmatrix(trainingMatrix)
    Y = trainingY
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(X,Y)
    log.warn ('scikit log_reg solution:%s '%log_reg.coef_)
    log.warn ('scikit log_reg intercept? %s'%log_reg.intercept_)
    log.warn ('model is trained now......')

    X = np.array(testMatrix)
    Y = testY
    c = log_reg.predict(X)  # or can I use score()
    print (c)
    print (Y)
    print (c-Y)
    print ('scikit total errors:', sum([abs(x) for x in (c-Y)]), 'out of',len(c))

if __name__ == "__main__":
    log.getLogger().setLevel(log.WARN)

    test_lady_gaga_gd5_1(kFeatures=50,maxRows=500,loops=2000)


