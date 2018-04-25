import time, itertools, os,requests, pandas, io
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import logging as log
import sympy as sp

#return in array with original result in [0], timing in [1]
def time_fn( fn, *args, **kwargs ):
    start = time.clock()
    results = fn( *args, **kwargs )
    end = time.clock()
    #fn_name = fn.__module__ + "." + fn.__name__
    #print fn_name + ": " + str(end-start) + "s"
    return [results,end-start]

#for data duplications
def churn(d, n):
    for _ in itertools.repeat(None, n):
        d = d.append(d)
    return d

#guess array formatter to 4d%f
def gf(guesses,scale=1):
    return ["{:0.4f}".format(g*scale) for g in guesses]

def setupBrainData(max=1000):
    if (os.path.isfile("myDataFrame.csv")):
        print 'reading cache copy from disk'
        return pandas.read_csv('myDataFrame.csv').head(max)    
    url='http://www.stat.ufl.edu/~winner/data/brainhead.dat'
    data=requests.get(url)
    col_names=('gender', 'age_range', 'head_size', 'brain_weight')
    col_widths=[(8,8),(16,16),(21-24),(29-32)]
    df=pandas.read_fwf(io.StringIO(data.text), names=col_names, colspec=col_widths)
    df.to_csv('myDataFrame.csv')
    return df.head(max)

#replicate/grow data
def makeFakeData():
    print('setup expanded datasets (dfs[])')
    df = setupBrainData()
    dfs = [df,churn(df,4),churn(df,8),churn(df,12),churn(df,16)]        
    for d in dfs:
        print (d.shape)
    return dfs

# x,y is string column from dataFrame to plot on x,y axes
def plotScatter(initData,xLabel,yLabel):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1,1,1)
    plt.ion()

    #scatter of test pts
    for _,row in initData.iterrows():
        x= row[xLabel]
        y= row[yLabel]
        ax.scatter(x, y)
    plt.pause(0.01)
    return ax

# plot new line Ax+B
def plotLine(ax,A,B,min=0,max=5000):
    if (len(ax.lines) > 0):
        ax.lines.pop()
    ax.plot([min,max],[min*A + B,max*A + B])
    plt.pause(0.01)
    return ax


# generic solver takes in hypothesis function, cost func, training matrix, theta array, yarray
# need to add params for stochastic/batchsize 
def grad_descent4(hFunc, cFunc, trainingMatrix, yArr, step=0.01, loop_limit=50, step_limit=0.00001, scale=1, batchSize=None):
    guesses = [0.01]*len(trainingMatrix[0])    # initial guess for all 
    costChange = 1.0
    if (batchSize == None):
        batchSize = len(trainingMatrix)  #@todo can i set this in func param

    # TODO do i really need these 2 here... pass them in?
    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array
    
    log.warn('init guesses %s',str(guesses))
    log.warn('init func: %s, training size: %d' %(str(hFunc),trainingMatrix.shape[0]))
    log.debug('ts: %s / xs: %s',ts,xs)

    costF = evalSumF2(cFunc,xs,trainingMatrix,yArr)  # cost fun evaluted for testData
    cost = 0.0+costF.subs(zip(ts,guesses))  
    log.warn('init cost: %f, costF %s',cost,str(costF)) # show first 80 char of cost evaluation

    trainingMatrix = shuffle(trainingMatrix, random_state=0)

    i=j=l=0
    while (abs(costChange) > step_limit and l<loop_limit):  # outer loop batch chunk
        i=j=k=0
        k = j+batchSize if j+batchSize<len(trainingMatrix) else len(trainingMatrix)
        dataBatch = trainingMatrix[j:k]

        while (i < len(trainingMatrix)/batchSize):  # inner batch size loop, min 1x loop
            for t,theta in enumerate(ts):
                pd = evalPartialDeriv2(cFunc,theta,ts,xs,dataBatch,guesses,yArr)
                guesses[t] = guesses[t] - step * pd
            previousCost = cost
            cost = costF.subs(zip(ts,guesses))
            costChange = previousCost-cost
            log.warn('l=%d,costChange=%f,cost=%f, guesses=%s'%(l, costChange,cost,gf(guesses,scale)))

            j = k
            k = j+batchSize if j+batchSize<len(trainingMatrix) else len(trainingMatrix)
            dataBatch = trainingMatrix[j:k]
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

