import time, itertools, os,requests, pandas, io
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error,log_loss
import logging as log
import sympy as sp
from myutils import *
import numpy as np


# generic solver takes in xEvalothesis function, cost func, training matrix, theta array, yarray
def grad_descent_sympy(hFunc, cFunc, trainingMatrix, yArr, step=0.01, loop_limit=50, step_limit=0.00001, batchSize=None):
    guesses = [0.01]*len(trainingMatrix[0])  # @TODO is this right or len(yArr)
    costChange = 1.0
    if (batchSize == None):
        batchSize = len(trainingMatrix)  #@todo can i set this in func param, or reduce to single expr
    batchSize = min(len(trainingMatrix),batchSize)

    # TODO do i really need these 2 here... pass them in?
    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array
    
    log.warn('init guesses %s'%(str(guesses)))
    log.warn('init func: %s, training size: %d' %(str(hFunc),trainingMatrix.shape[0]))
    log.debug('ts: %s / xs: %s',ts,xs)

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
    cost = 0.0+costF.subs(zip(ts,guesses))  
    log.warn('init cost: %f, costF %s',cost,str(costF)) # show first 80 char of cost evaluation

    trainingMatrix = shuffle(trainingMatrix, random_state=0)   
    yArr = shuffle(yArr, random_state=0)   

    i=j=l=0
    while (abs(costChange) > step_limit and l<loop_limit):  # outer loop batch chunk
        i=j=k=0
        k = j+batchSize if j+batchSize<len(trainingMatrix) else len(trainingMatrix)
        dataBatch = trainingMatrix[j:k]
        yBatch = yArr[j:k]
        log.debug('outer - batch %d, j: %d, k: %d'%(len(dataBatch),j,k))

        while (i < len(trainingMatrix)/batchSize):  # inner batch size loop, min 1x loop
            for t,theta in enumerate(ts):
                pd = evalPartialDeriv2(cFunc,theta,ts,xs,dataBatch,guesses,yBatch)
                guesses[t] = guesses[t] - step * pd
            previousCost = cost
            cost = costF.subs(zip(ts,guesses))
            costChange = previousCost-cost
            if l % 50 == 0:
                log.warn('l=%d,bs=%d,costChange=%f,cost=%f, guesses=%s'%(l,batchSize, costChange,cost,gf(guesses)))
            j = k
            k = j+batchSize if j+batchSize<len(trainingMatrix) else len(trainingMatrix)
            dataBatch = trainingMatrix[j:k]
            yBatch = yArr[j:k]
            i += 1
            l += 1
    return guesses

# expnd to avg(sum(f evaluated for xs,testData,yarr)) 
def evalSumF2(f,xs,trainingMatrix,yArr):  # @TODO change testData to matrix
    assert (len(xs) == len(trainingMatrix[0]))
    assert (len(trainingMatrix) == len(yArr))
    n=0.0
    _f = f 
    for i,row in enumerate(trainingMatrix):
        for j,x in enumerate(xs):
            _f = _f.subs(x,row[j])
        n+= _f.subs(sp.symbols('y'),yArr[i])
        log.debug('_f: %s n: %s',_f, n)
        _f = f
        log.info('------ expand: y:(%d) %s to %s ', yArr[i],str(row),str(n))
    n *= (1.0/len(trainingMatrix))
    log.info('f (%d) %s - \n  ->expanded: %s '%(len(trainingMatrix[0]), str(f), str(n)))
    return n 

# generate deriv and sub all x's w/ training data and theta guess values
def evalPartialDeriv2(f,theta,ts,xs,trainingMatrix,guesses,yArr):
    pdcost = evalSumF2(sp.diff(f,theta),xs,trainingMatrix,yArr)
    pdcost = pdcost.subs(zip(ts,guesses))
    log.info ('    --> pdcost %f ;  %s  ;  %s: '%(pdcost,str(f),str(theta)))
    return pdcost

#reference impl for mean_square cost see: https://stackoverflow.com/questions/47795918/logistic-regression-gradient-descent
def grad_descent_linr_mse(xMatrix, yArr, step_limit, step):
    m = len(xMatrix) # number of samples
    theta = [0.01]*len(xMatrix[0]) # init guesses
    x_transpose = xMatrix.transpose()
    for iter in range(0, step_limit):
        hypothesis = np.dot(xMatrix, theta)
        error = hypothesis - yArr   # error/cost
        J = np.sum(error ** 2) * (2.0 / m)  # sum of errors (total cost)
        gradient = np.dot(x_transpose, error) * (2.0/m)         
        theta = theta - step * gradient  # update
        log.warn("iter %s | J: %.3f | theta %s grad %s" % (iter, J, theta, gradient))
    return theta

# reference impl for mean_square cost see: https://stackoverflow.com/questions/47795918/logistic-regression-gradient-descent
# annoying Y has to be a np.array.T columns... 
def grad_descent_logr(X,Y,iterations=500, learning_rate=0.5):
    if (Y.ndim != 2):
        Y = Y.reshape([-1,1])

    def gradient_Descent(theta, alpha, x , y):
        m = x.shape[0]
        h = sigmoid(np.matmul(x, theta))
        grad = np.matmul(X.T, (h - y)) * 1.0 / m
        theta = theta - alpha * grad
        return theta

    def cost(x, y, theta):
        m = x.shape[0]
        h = sigmoid(np.matmul(x, theta))
        cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1 -y.T), np.log(1 - h)))/m
        return cost

    Theta = np.array([[0], [0]])

    for i in range(iterations):
        Theta = gradient_Descent(Theta, learning_rate, X, Y)
        if i % 50 == 0:
            log.warn('i %s  thetas %s  cost %s'%(i,gf(Theta), gf(cost(X, Y, Theta)[0])))
    return Theta

# compare using standard scikit learn logistic regression 
def sklearn_logr_comp(X,Y):
    from sklearn.linear_model import LogisticRegression
    l_reg = LogisticRegression()
    l_reg.fit(X,Y)
    return l_reg.coef_, l_reg.intercept_

def sklearn_linr_comp(X,Y):
    from sklearn.linear_model import LinearRegression
    l_reg = LinearRegression()
    l_reg.fit(X,Y)
    return l_reg.coef_, l_reg.intercept_

# generic solver, errorFunc(y,x), costFunc(y,x), xArr (np.array), yArr (np.array)
def grad_descent5(eFunc, cFunc, xArr, yArr, step=0.01, loop_limit=50, step_limit=0.00001, batchSize=None):
    batchSize = len(xArr) if batchSize == None else min(len(xArr),batchSize)
    guesses = [0.01]*len(xArr[0])  # initial guess for all 
    costChange = costSum = 1.0
    xArr,yArr = shuffle(xArr, random_state=0), shuffle(yArr, random_state=0)  # perhaps should shuffle inside outer loop
    
    i=j=l=0
    while (abs(costChange) > step_limit and l<loop_limit):  # outer loop batch chunk
        i=j=k=0
        k = j+batchSize if j+batchSize<len(xArr) else len(xArr)
        xBatch,yBatch = xArr[j:k],yArr[j:k]

        while (i < len(xArr)/batchSize):  # inner batch size loop, min 1x loop
            previousCost = costSum
            xEval = np.dot(xBatch, guesses)  # array of X*0 evaluations
            error = eFunc(yBatch, xEval)     # errorFunc could be x-y or sig(x)-y   
            costSum = cFunc(yBatch,error+yBatch)  # cFunc log_loss, or mean_error_square, or custom.  xEval = error+yBatch in case of sig(x)
            guesses = guesses - step * np.dot(xBatch.T, error) * 2.0/len(xBatch)  # ng std formula: 0 := 0 - a/m * X.T*(g(X*0) - Y)
            costChange = previousCost-costSum
            if l % 50 == 0:
                log.warn('l=%d,i=%d,bs=%d,costChange=%f,cost=%f, guesses=%s'%(l,i,batchSize, costChange,costSum,gf(guesses)))
            j = k
            k = j+batchSize if j+batchSize<len(xArr) else len(xArr)
            xBatch,yBatch = xArr[j:k],yArr[j:k]
            i += 1
            l += 1
    return guesses

# replaced by log_loss scikit function, this one is easier because it doesn't need both 0's and 1's...
def sigmoidCost(y,x):
    ret=[]
    for i,x_ in enumerate(x):
        p = sigmoid(x_) 
        cost = -y[i]*np.log(p) - (1-y[i])*np.log(1.0-p)
        ret.append(cost)
    return sum(ret)/len(y)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

## test section
if __name__ == "__main__":
    trainingMatrix = np.array([[0,10],[1,12],[10,5],[12,3]])  # 2 features
    yArr = [1,1,0,0]
    guesses = [0.01]*len(trainingMatrix[0])
    xEval = trainingMatrix.dot(guesses)
    err = log_loss(yArr, xEval)
    log.warn('init err/cost: %f'%err)

#    gs = grad_descent5(lambda y,x: sigmoid(x)-y,log_loss,trainingMatrix,yArr,step=0.1,step_limit=0.000001,loop_limit=1000, batchSize=4)    
#    log.warn('final: %s'%gs)

    Y2=np.array(yArr).reshape([-1, 1])
    gs = grad_descent_logr(trainingMatrix, Y2, 500, 0.5)
    log.warn('final: %s'%gs)

    X = np.asmatrix(trainingMatrix)
    Y = yArr
#    log.warn ('target Linear Reg sol: %s'% str((X.T.dot(X)).I.dot(X.T).dot(Y)))

    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression()
    log_reg.fit(X,Y)
    print ('scikit log_reg solution:',gf(log_reg.coef_[0]))
    print ('scikit log_reg intercept?',log_reg.intercept_)

#    assert(round(gs[0],1) == round(log_reg.coef_[0][0],1))
#    assert(round(gs[1],1) == round(log_reg.coef_[0][1],1))

