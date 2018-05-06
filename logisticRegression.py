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

# take in x-nparray matrix, and return reduced matrix and features
def reduceFeatures(X, Y, features, kFeatures):
    # SelectKBest 
    log.warn('feature reduce KBest: %s'%(str(X.shape)))
    model = SelectKBest(chi2, k=kFeatures)
    X_new = model.fit_transform(X, Y)   # need to keep features
    log.warn('KBest %d applied %s'%(kFeatures, str(X_new.shape)))
    df = pandas.DataFrame(X_new)
    indexes = model.get_support(True)
    words = [features[p] for p in indexes]

    # reduce to K features
    df = pandas.DataFrame(X, columns=features)
    df = df[words]
    X_new = df.as_matrix()

    return X_new,words

# test with mike eng's dataset
def testGagaClassifier(kFeatures,maxRows,loops):
    print (inspect.currentframe().f_code.co_name,kFeatures)
    xMatrix,yArr,features,fnames = fe.getGagaData(maxrows=maxRows,stopwords='english')

    xMatrix = shuffle(xMatrix, random_state=0)   
    yArr = shuffle(yArr, random_state=0) 

    partition = int(.70*len(yArr))

    trainingMatrix = xMatrix[:partition]
    trainingY = yArr[:partition]
    testMatrix = xMatrix[partition:]
    testY = yArr[partition:]

    X = np.array(trainingMatrix)
    Y = trainingY
    X,rfeatures = reduceFeatures(X, Y, features, kFeatures)
    m1,c1 = fe.countWords2(X, features, fnames)
    log.warn('reduced matrix: %s'%str(m1)) 
    gs = grad_descent5(lambda y,x: sigmoid(x)-y,sigmoidCost,X,Y,step=0.1,step_limit=0.000000001,loop_limit=loops)
    log.error('mymodel guesses: %s len %d'%(gf(gs), len(gs)))

    # reduce test set similarly (note below works because we know full set of train+test features ahead of time)    
    df = pandas.DataFrame(testMatrix, columns=features)   # new df w/ column names
    X = df[rfeatures].as_matrix()                # filter out only rfeatures

    testRes = np.dot(X, gs)
    testResRound = [round(sigmoid(x),0) for x in testRes]
    testDiffs = np.array(testResRound) - np.array(testY)
    log.warn ('raw results %s '%(gf(testRes)))
    log.warn ('sig results %s'%gf([sigmoid(x) for x in testRes]))
    log.warn ('0|1 results %s'%([round(sigmoid(x)) for x in testRes]))
    log.warn (testDiffs)
    log.error ('mymodel errors: %s / %s = %f'%(sum([abs(x) for x in testDiffs]),len(testY),sum([abs(x) for x in testDiffs])/len(testY)))

    # sklearn validation
    X = np.asmatrix(trainingMatrix)
    X,rfeatures = reduceFeatures(X, Y, features, kFeatures)
    Y = trainingY
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(X,Y)
    log.error ('scikit log_reg solution:%s len %d '%(gf(log_reg.coef_[0]),len(log_reg.coef_[0])))
    log.warn ('scikit log_reg intercept? %s'%log_reg.intercept_)
    log.warn ('model is trained now......')

    X = np.array(testMatrix)
    df = pandas.DataFrame(testMatrix, columns=features)   # new df w/ column names
    X = df[rfeatures].as_matrix()                # filter out only rfeatures
    Y = testY
    
    c = log_reg.predict(X)  # or can I use score()
    log.info (c)
    log.info (Y)
    log.info (c-Y)
    log.error ('scikit total errors: %s / %s = %f'%(sum([float(abs(x)) for x in (c-Y)]),len(c),sum([float(abs(x)) for x in (c-Y)])/len(Y)))

if __name__ == "__main__":
    log.getLogger().setLevel(log.ERROR)

    testGagaClassifier(kFeatures=10,maxRows=500,loops=2550)
    testGagaClassifier(kFeatures=100,maxRows=500,loops=2550)
    testGagaClassifier(kFeatures=1000,maxRows=500,loops=2550)


