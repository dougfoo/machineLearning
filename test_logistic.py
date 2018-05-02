# content of test_sample.py
# test test
from myutils import *
import featureEngineering as fe
import pandas, inspect
from mpmath import *
import numpy as np
import logging as log

def inc(x):
    print (inspect.currentframe().f_code.co_name)
    return x + 1

def test_inc():
    print (inspect.currentframe().f_code.co_name)
    assert(4 == inc(3))

def test_lr_gaga_solver_1():
    print (inspect.currentframe().f_code.co_name)
    yArr = [1,0]
    trainingMatrix = np.array([[4,0],[0,5]])  #dummy data 1's, word1, word2 -- first row gaga, 2nd non

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    f = (sp.Matrix(ts).T * sp.Matrix(xs) ) [0]
    g = 1 / (1+mp.e**-f)   # wrap in sigmoid
    y = sp.symbols('y')
    cFunc = (0-y)*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData

    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.01,step_limit=0.00001,loop_limit=50)    
    assert (round(gs[0],2) == 0.33)
    assert (round(gs[1],2) == -0.34)

def test_lr_gaga_solver_2():
    print (inspect.currentframe().f_code.co_name)
    yArr = [1,1,0,0]
    trainingMatrix = np.array([[12,5,1,2],[10,5,3,1],[0,1,8,2],[2,1,7,7]])  #dummy data 1's, word1, word2 -- first row gaga, 2nd non

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    f = (sp.Matrix(ts).T * sp.Matrix(xs) ) [0]
    g = 1 / (1+mp.e**-f)   # wrap in sigmoid
    y = sp.symbols('y')
    cFunc = (0-y)*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData

    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.01,step_limit=0.00001,loop_limit=10,batchSize=2)    
    assert (round(gs[0],2) == 0.15)
    assert (round(gs[1],2) == 0.05)
    assert (round(gs[2],2) == -0.12)
    assert (round(gs[3],2) == -0.06)

def test_lr_gaga_solver_3():
    print (inspect.currentframe().f_code.co_name)
    yArr = np.array([1,1,0,0,0])
    trainingMatrix = np.array([[12,5,1,2,1],[10,5,3,1,2],[0,1,8,2,1],[2,1,7,7,9]])  #dummy data 1's, word1, word2 -- first row gaga, 2nd non

    gs = grad_descent5(lambda y,x: sigmoid(x)-y,sigmoidCost,trainingMatrix,yArr,step=0.01,step_limit=0.00001,loop_limit=100,batchSize=3)    
    gs1 = grad_descent5(lambda y,x: sigmoid(x)-y,log_loss,trainingMatrix,yArr,step=0.01,step_limit=0.00001,loop_limit=100,batchSize=3)    
    log.warn('final: %s'%gs)
    log.warn('final: %s'%gs1)
    assert (round(gs[0],2) == round(gs1[0],2))
    assert (round(gs[1],2) == round(gs1[1],2))
    assert (round(gs[2],2) == round(gs1[2],2))

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

def testLRGaga2(kFeatures=50,ts=10):
    print (inspect.currentframe().f_code.co_name)
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
    print(df.describe())
    print(df)
    print(yArr)
    trainingMatrix = df.as_matrix()
    labels = pickwords
 
    m1,c1 = fe.countWords2(trainingMatrix, labels, fnames)

    gs = grad_descent5(lambda y,x: sigmoid(x)-y,sigmoidCost,trainingMatrix,yArr,step=0.01,step_limit=0.00001,loop_limit=100)    
    log.warn('guesses: %s', gf(gs))
    log.warn('reduced matrix: %s'%str(m1)) 


if __name__ == "__main__":
    log.getLogger().setLevel(log.WARN)

    # test_lr_gaga_solver_1()
    # test_lr_gaga_solver_2()
    #test_lr_gaga_solver_3()
    testLRGaga2(kFeatures=50,ts=100)



