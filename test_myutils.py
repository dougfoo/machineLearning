# test test
from myutils import *

def setup_module(module):
    print ("setup_module      module:%s" % module.__name__)
 
def teardown_module(module):
    print ("teardown_module   module:%s" % module.__name__)
 
def setup_function(function):
    log.basicConfig(level=log.ERROR) 
    print ("setup_function    function:%s" % function.__name__)
 
def teardown_function(function):
    print ("teardown_function function:%s" % function.__name__)

def test_dummy():
    assert (1+3) == 4

def test_evalSumF2():
    x0,x1,t0,t1,y = sp.symbols('x0,x1,t0,t1,y')
    f = (x0*t0 + x1*t1 - y)**2  # sum-square-error (MSE) of mx+b
    xs = [x0,x1]
    trainingMatrix = [[1,5],[1,10]]
    f = sp.diff(f,x0)
    s = evalSumF2(f,xs,trainingMatrix,[10,15])
    assert(str(s) == "1.0*t0*(t0 + 5*t1 - 10) + 1.0*t0*(t0 + 10*t1 - 15)")

def test_evalSumF2_1():
    x0,x1,t0,t1,y = sp.symbols('x0,x1,t0,t1,y')
    f = (x0*t0 + x1*t1 - y)**2  # sum-square-error (MSE) of mx+b
    xs = [x0,x1]
    trainingMatrix = [[1,2],[1,4]]
    log.warn (f)
    f = sp.diff(f,x1)
    log.warn (f)
    s = evalSumF2(f,xs,trainingMatrix,[10,11])
    log.warn (s)
    assert(str(s) == "1.0*t1*(t0 + 2*t1 - 10) + 1.0*t1*(t0 + 4*t1 - 11)")

def test_evalPartialDeriv2():
    df = setupBrainData(4)  # 'gender', 'age_range', 'head_size', 'brain_weight'
    df['bias'] = 1
    trainingMatrix = df[['bias','head_size']]
    trainingMatrix = trainingMatrix.as_matrix()
    yArr = df[['brain_weight']].as_matrix()
    guesses = [0*len(yArr)]

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    log.warn (trainingMatrix)
    log.warn (ts)
    log.warn (xs)

    y = sp.symbols('y')
    f = ts[0]*xs[0] + ts[1]*xs[1]
    cFunc = (f - y)**2  # error squared
    
    log.warn('init guesses %s',str(guesses))
    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.debug('ts: %s / xs: %s',ts,xs)

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
    cost = 0.0+costF.subs(zip(ts,guesses))  
    log.warn ('costF ',costF)
    log.warn ('cost ',cost)
    
    pds = []
    for theta in ts:
        pd = evalPartialDeriv2(f,theta,ts,xs,trainingMatrix,guesses,yArr)
        log.warn ('deriv %s of %s = %f'%(str(f),str(theta),pd))
        pds.append(pd)

    assert(pds[0] == 1.0)
    assert(pds[1] == 4072.0)

def test_grad_descent4():
    df = setupBrainData(4)  # 'gender', 'age_range', 'head_size', 'brain_weight'
    log.warn(df.describe())
    scale1=4600
    df['bias'] = 1
    df['head_size'] = df['head_size']/scale1
    df['brain_weight'] = df['brain_weight']/scale1
    trainingMatrix = df[['bias','head_size']]
    trainingMatrix = trainingMatrix.as_matrix()
    yArr = df[['brain_weight']].as_matrix()

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array
    y = sp.symbols('y')
    f = ts[0]*xs[0] + ts[1]*xs[1]
    cFunc = (f - y)**2  # error squared
    guesses = [0.01*len(yArr)]

    log.warn('init guesses %s',str(guesses))
    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
    cost = 0.0+costF.subs(zip(ts,guesses))  
    log.warn('costF %s'%(str(costF)))
    log.warn('cost %s'%(str(cost)))

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.001,loop_limit=1000,step_limit=0.0000001)    
    log.warn('scaled A: %f'%(gs[0]*scale1))
    log.warn('scaled B: %f'%(gs[1]*scale1))

    assert(gs[0] == 0.920911)
    assert(gs[1] == 1.048603)    


def test_grad_descent4_1():
# test hack
    import numpy as np
    trainingMatrix = np.array([[1,1],[1,2]])
    yArr = [2,3]
    guesses = [0.01*len(yArr)]

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    y = sp.symbols('y')
    f = ts[0]*xs[0] + ts[1]*xs[1]
    cFunc = (f - y)**2  # error squared
    
    log.warn('init guesses %s',str(guesses))
    log.error('init func: %s, training size: %d' %(str(f),len(trainingMatrix)))
    log.warn('ts: %s / xs: %s',ts,xs)

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
    cost = 0.0+costF.subs(zip(ts,guesses))  
    log.warn('costF %s'%(str(costF)))
    log.warn('cost %s'%(str(cost)))

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.05,loop_limit=1000)    
    log.warn('scaled A: %f'%(gs[0]))
    log.warn('scaled B: %f'%(gs[1]))

    assert(round(gs[0],2) == 0.92)
    assert(round(gs[1],2) == 1.05)    

def test_grad_descent4_2():
    import numpy as np
    trainingMatrix = np.array([[1,4],[1,10],[1,20]])
    yArr = [8,18,42]
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

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.005,loop_limit=1000)    
    log.warn('scaled A: %f'%(gs[0]))
    log.warn('scaled B: %f'%(gs[1]))

    X = np.asmatrix(trainingMatrix)
    Y = yArr
    log.warn ('target sol: %s'% str((X.T.dot(X)).I.dot(X.T).dot(Y)))

    assert(round(gs[0],2) == -1.59)
    assert(round(gs[1],2) == 2.14)    

def test_grad_descent4_3():
    import numpy as np
    trainingMatrix = np.array([[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8]])
    yArr = [14,16,18,20,21,22,22]
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

    gs = grad_descent4(f,costF,trainingMatrix,yArr,step=0.01,loop_limit=1000, batchSize=1)    
    log.warn('scaled A: %f'%(gs[0]))
    log.warn('scaled B: %f'%(gs[1]))

    X = np.asmatrix(trainingMatrix)
    Y = yArr
    log.warn ('target sol: %s'% str((X.T.dot(X)).I.dot(X.T).dot(Y)))

    assert(round(gs[0]) == 12)
    assert(round(gs[1],1) == 1.5)    

log.basicConfig(level=log.WARN)

#test_evalSumF2()
#test_evalSumF2_1()
#test_evalPartialDeriv2()
#test_grad_descent4_1()
#test_grad_descent4_2()
test_grad_descent4_3()
#test_grad_descent4()

