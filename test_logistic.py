# content of test_sample.py
# test test
from myutils import *
import featureEngineering as fe
import pandas
import numpy as np

def inc(x):
    return x + 1

def test_inc():
    assert(4 == inc(3))

def test_gaga_solver(kFeatures=50,bs=4,ts=10):
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

    guesses = [0.01*len(yArr)]

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array
    xt = sp.Matrix(ts).T * sp.Matrix(xs)
    f = xt[0]
    y = sp.symbols('y')
    cFunc = (f - y)**2  # error squared

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
#    cost = 0.0+costF.subs(zip(ts,guesses))  

    log.warn('init guesses %s',str(guesses))
    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.001,step_limit=0.00001,loop_limit=500, batchSize=bs)    
    log.warn('scaled A: %f'%(gs[0]))
    log.warn('scaled B: %f'%(gs[1]))

    X = np.asmatrix(trainingMatrix)
    Y = yArr
    sol = (X.T.dot(X)).I.dot(X.T).dot(Y)
    sol = np.asarray(sol)[0]
    log.warn ('target sol (X.T * X)^-1 * X.T*Y:     %s'% str(fe.gf(sol)))

#    assert(round(gs[0]) == 11)
#    assert(round(gs[1],1) == 1.5)    

log.basicConfig(level=log.WARN)
t = time_fn(test_gaga_solver,20,4,100)
t2 = time_fn(test_gaga_solver,40,4,100)

print (t)
print (t2)


