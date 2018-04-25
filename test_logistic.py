# content of test_sample.py
# test test
from myutils import *
from featureEngineering import *

import numpy as np

def inc(x):
    return x + 1

def test_inc():
    assert(4 == inc(3))

def test_gaga_solver():
    trainingMatrix1,yArr,labels,fnames = getGagaData(maxrows=4)
    t1 = numpy.array(trainingMatrix1)

    # SelectKBest
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X, y = t1, yArr
    print(X.shape)
    model = SelectKBest(chi2, k=50)
    X_new = model.fit_transform(X, y)   # need to keep labels
    print('KBest',X_new.shape)
    df = pandas.DataFrame(X_new)
    print (df.head())
    picklist = model.get_support(True)
    pickwords = [labels[p] for p in picklist]

    guesses = [0.01*len(yArr)]

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    y = sp.symbols('y')
    f = ts[0]*xs[0] + ts[1]*xs[1]
    cFunc = (f - y)**2  # error squared

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
    cost = 0.0+costF.subs(zip(ts,guesses))  
    log.warn('costF %s'%(str(costF)))
    log.warn('cost %s'%(str(cost)))

    log.warn('init guesses %s',str(guesses))
    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.01,loop_limit=1000, batchSize=bs)    
    log.warn('scaled A: %f'%(gs[0]))
    log.warn('scaled B: %f'%(gs[1]))

    X = np.asmatrix(trainingMatrix)
    Y = yArr
    log.warn ('target sol: %s'% str((X.T.dot(X)).I.dot(X.T).dot(Y)))

    assert(round(gs[0]) == 11)
    assert(round(gs[1],1) == 1.5)    


log.basicConfig(level=log.WARN)
test_gaga_solver()
