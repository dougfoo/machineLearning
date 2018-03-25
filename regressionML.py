import requests, pandas, io
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *

#return in array with original result in [0], timing in [1]

def setupData(max=1000):
    url='http://www.stat.ufl.edu/~winner/data/brainhead.dat'
    data=requests.get(url)
    col_names=('gender', 'age_range', 'head_size', 'brain_weight')
    col_widths=[(8,8),(16,16),(21-24),(29-32)]
    df=pandas.read_fwf(io.StringIO(data.text), names=col_names, colspec=col_widths)
    return df.head(max)

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

##########################
##### test runnners  #####

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


def plotGradientRun():
    import pandas as pd
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.ion()

    #scatter of test pts
    df2 = setupData()
    for _,row in df2.iterrows():
        x= row['head_size']
        y= row['brain_weight']
        ax.scatter(x, y)
    max = df2['head_size'].max()
    min = df2['head_size'].min()
    plt.pause(1)

    # retrace gradient descent
    df = pd.read_csv('run.txt', header=None)
    print(df.shape)
    print(df.head())

    for _,d in df.iterrows(): 
        m = float(d[2].split('=')[1])
        b = float(d[3].split('=')[1])
        ax.plot([min,max],[min*m + b,max*m + b])
        plt.pause(0.1)
        ax.lines.pop()

    while True:
        plt.pause(0.05)

plotGradientRun()
#testLD2()
#testPlot()
    