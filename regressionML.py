import requests, pandas, io
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
import itertools
import time

#return in array with original result in [0], timing in [1]
def time_fn( fn, *args, **kwargs ):
    start = time.clock()
    results = fn( *args, **kwargs )
    end = time.clock()
    fn_name = fn.__module__ + "." + fn.__name__
    #print fn_name + ": " + str(end-start) + "s"
    return [results,end-start]

def setupData(max=1000):
    url='http://www.stat.ufl.edu/~winner/data/brainhead.dat'
    data=requests.get(url)
    col_names=('gender', 'age_range', 'head_size', 'brain_weight')
    col_widths=[(8,8),(16,16),(21-24),(29-32)]
    df=pandas.read_fwf(io.StringIO(data.text), names=col_names, colspec=col_widths)
    return df.head(max)

def churn(d, n):
    for _ in itertools.repeat(None, n):
        d = d.append(d)
    return d

def makeFakeData():
    print('setup expanded datasets (dfs[])')
    df = setupData()
    dfs = [df,churn(df,4),churn(df,8),churn(df,12),churn(df,16)]        
    for d in dfs:
        print (d.shape)
    return dfs

# evaluate/calculate f with data sub for x and y (very slow iterative)
def evalSumF(f,x,y,testData):
    n=0
    for _,d in testData.iterrows():  # global test data
        n += f.subs(x,d.head_size).subs(y,d.brain_weight)
    return n * (1.0/len(testData))

# generate partial derivative of f, with respect to v, for testData (x,y) and evaluate
def evalPartialDeriv(f,x,y,testData,v,guessV,o,guessO):
    pc = evalSumF(sp.diff(f,v),x,y,testData)
    pceval = pc.subs(v,guessV).subs(o,guessO)
    #print ('    v,p,pc:pceval',v,guessV,p,pc,pceval)
    return pceval

# semi-hard coded solver for f(x,y) given data series testData, start w/ guess, solve cost, iterate cost+/-partialDerivs
def grad_descent2(f, testData=setupData()):
    guessA = guessB = 1.0   #initial guess y=1x+1

    stepA = 0.00000005   #dif step for diff A,B ?
    stepB = 0.25         #maybe normalize data first
    step_limit = 0.0001  # when to stop, when cost stops changing
    loop_limit = 2000    # arbitrary max limits
    costChange = 1.0

    A,B,x,y = sp.symbols('A B x y')
    e = (f - y)**2  # error squared
    print ('init guess A: %f, B: %f'%(guessA,guessB))
    print ('init func: %s, test size: %d' %(str(f),testData.shape[0]))
    costF = evalSumF(e,x,y,testData)  # cost fun evaluted for testData
    print('init costF',str(costF)[:80])
    costEval = costF.subs(A,guessA).subs(B,guessB)  # cost evaluted for A B guess
    print('init cost',costEval)

    i=0  
    while (abs(costChange) > step_limit and i<loop_limit):  # arbitrary limiter
        pda = evalPartialDeriv(e,x,y,testData,A,guessA,B,guessB)
        pdb = evalPartialDeriv(e,x,y,testData,B,guessB,A,guessA)
        guessA = guessA - stepA * pda
        guessB = guessB - stepB * pdb
        previousCost = costEval
        costEval = costF.subs(A,guessA).subs(B,guessB)
        costChange = previousCost-costEval
        print ('i=%d,cost=%d,A=%f,B=%f'%(i, int(costEval), guessA, guessB))
        i=i+1
    return guessA,guessB

# matrix method for gradient descent
def grad_descent3(x,y):
    guessA = guessB = 1.0
    return guessA,guessB

def testLD2():
    timings = []
    dfs = makeFakeData()
    A,B,x = sp.symbols('A B x')
    f = A*x + B  # linear func y=mx+b

    for d in dfs[0:2]:
        r = time_fn(grad_descent2,f,d)
        print ('finished for rows,time(s)',d.shape[0], r[1])
        timings.append(r)
    print('*** done')
    print(timings)

#test matrix
def testLD3():
    timings = []
    dfs = makeFakeData()
    x=[]
    y=[]

    for d in dfs[0:2]:
        r = time_fn(grad_descent3,x,y)
        print ('finished for rows,time(s)',d.shape[0], r[1])
        timings.append(r)
    print('*** done')
    print(timings)

def testPlot():
    import matplotlib.pyplot as plt

    df = setupData(10)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

#    plt.axis([0, 2000, 0, 5000])
#    plt.ion()


    for _,row in df.iterrows():
        i= row['brain_weight']
        y= row['head_size']
        ax.scatter(i, y)
        plt.pause(0.005)

    ax.plot([0,1],[0,1], transform=plt.gca().transAxes)
    ax.plot([0,1],[0,3], transform=plt.gca().transAxes)
    plt.pause(1)
    
    print (ax.lines)

    guesses = []
    for g in guesses:
        plt.plot([0,1],[0,1], transform=plt.gca().transAxes)
        plt.pause(0.1)
        plt.clf()
#    plt.plot(1,2)
#    plt.plot(1000,2000)
#

    while True:
        plt.pause(0.05)

testPlot()
    