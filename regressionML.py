import requests, pandas, io
import matplotlib.pyplot as plt
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
import argparse

# @todo visuals/demo 
#      - make it run command line w/ parameters 
#      - add plot titles
#      - autosize plot better 
#      - plot diff of algo speed / efficiency
# @todo algo fixes
#      - normalize/scale x/y better
#      - change from A,B to Theta[n] for parameters to generalize 

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

# semi-hard coded batch solver for f(x,y) given data series testData, start w/ guess, solve cost, iterate cost+/-partialDerivs
def grad_descent2(f, testData=setupData(), pltAx=False, batchSize=None, t='grad_desc'):
    guessA = guessB = 1.0   #initial guess y=1x+1

    stepA = 0.00000005   #dif step for diff A,B ?
    stepB = 0.25         #maybe normalize data first
    step_limit = 100.0   # when to stop, when cost stops changing
    loop_limit = 500     # arbitrary max limits
    costChange = step_limit+1

    A,B,x,y = sp.symbols('A B x y')
    e = (f - y)**2  # error squared
    print ('init guess A: %f, B: %f'%(guessA,guessB))
    print ('init func: %s, test size: %d' %(str(f),testData.shape[0]))
    costF = evalSumF(e,x,y,testData)  # cost fun evaluted for testData
    print('init costF',str(costF)[:80])  # oddly this line crashes on yoga tablet
    costEval = costF.subs(A,guessA).subs(B,guessB)  # cost evaluted for A B guess
    print('init cost',costEval)

    # add optional plot - scatter of testData
    if (pltAx):
        pltAx=plotScatter(testData,xLabel='head_size',yLabel='brain_weight')
        pltAx.set_title(t)
        max = testData['head_size'].max()
        min = testData['head_size'].min()
        print(pltAx, max,min)

    i=j=l=0
    if (batchSize == None):
        batchSize = len(testData)  #@todo can i set this in func param

    # outer loop std grad descent solver loop
    while (abs(costChange) > step_limit and l<loop_limit):  # arbitrary limiter
        i=j=k=0
        testData = shuffle(testData)
        k = j+batchSize if j+batchSize<len(testData) else len(testData)
        dataBatch = testData[j:k]
#        print('shuffled loop - batch size: %d, j: %d, k: %d, %d'%(batchSize,j,k, len(testData)/batchSize))

        # inner batch of size batchSize - test in batches and redo again
        while (i < len(testData)/batchSize):
            pda = evalPartialDeriv(e,x,y,dataBatch,A,guessA,B,guessB)
            pdb = evalPartialDeriv(e,x,y,dataBatch,B,guessB,A,guessA)
            guessA = guessA - stepA * pda
            guessB = guessB - stepB * pdb
            previousCost = costEval
            costEval = costF.subs(A,guessA).subs(B,guessB)
            costChange = previousCost-costEval
            print ('t=%s,l=%d,i=%d,cost=%d,A=%f,B=%f,bs=%d'%(t, l, i, int(costEval), guessA, guessB, batchSize))
            # add optional plot of current regression line
            if (pltAx):
                plotLine(pltAx,guessA,guessB,min,max)
            j = k
            k = j+batchSize if j+batchSize<len(testData) else len(testData)
            dataBatch = testData[j:k]
            i += 1
            l += 1

    return guessA,guessB

##########################
##### test runnners  #####

# test normal gradient descent
def testGD(plt=False, gd=grad_descent2, bs=None, ts=None, t=None):
    d = setupData(ts)
    A,B,x = sp.symbols('A B x')
    f = A*x + B  # linear func y=mx+b

    timing = time_fn(gd,f,d,plt,bs,t)
    print ('finished for rows,time(s)',d.shape, timing)
    print('*** done')
    print(timing)
    return timing

# test plotting from file
def plotGradientRun():
    import matplotlib.pyplot as plt

    plt.xlim(2000,5000)
    plt.ylim(-700,4000)
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
    df = pandas.read_csv('run.txt', header=None)
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

##################
## runtime main ##
##################

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-bs', help='batch size', type=int)
parser.add_argument('-ts', help='training size', type=int)
parser.add_argument('-t', help='title', type=str)
args = vars(parser.parse_args())

_bs = args['bs']
_ts = args['ts']
_t = args['t']

#plotGradientRun()
t1 = testGD(plt=True, gd=grad_descent2, bs=_bs, ts=_ts, t=_t)   # equivalent of stocastic descent
#t2 = testGD(plt=True, gd=grad_descent2, bs=5, ts=52)  # equavalent of mini-batch descent
#t3 = testGD(plt=True, gd=grad_descent2, ts=52)         # standard batch descent

#t4 = testGD(plt=True, gd=grad_descent2, bs=1, ts=200)   # equivalent of stocastic descent
#t5 = testGD(plt=True, gd=grad_descent2, bs=20, ts=200)  # equavalent of mini-batch descent
#t6 = testGD(plt=True, gd=grad_descent2, ts=200)         # standard batch descent

print('timing & results - stocastic:', t1)
#print('timing & results - mini batch:', t2)
#print('timing & results - batch:', t3)

#print('timing & results - stocastic:', t4)
#print('timing & results - mini batch:', t5)
#print('timing & results - batch:', t6)
