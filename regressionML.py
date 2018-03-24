import requests, pandas, io
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
import itertools
import time

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

# generate partial derivative of e, with respect to v, for testData (x,y) and evaluate
def evalPartialDeriv(e,x,y,testData,v,guessV,o,guessO):
    pc = evalSumF(sp.diff(e,v),x,y,testData)
    pceval = pc.subs(v,guessV).subs(o,guessO)
    #print ('    v,p,pc:pceval',v,guessV,p,pc,pceval)
    return pceval

# hard coded solver, start w/ guess, solve cost, iterate cost+/-partialDerivs
def grad_descent2(testData=setupData()):
    guessA = guessB = 1.0   #initial guess y=1x+1

    stepA = 0.00000005   #dif step for diff A,B ?
    stepB = 0.25         #maybe normalize data first
    step_limit = 0.0001  # when to stop, when cost stops changing
    loop_limit = 2000    # arbitrary max limits
    costChange = 1.0

    A,B,x,y = sp.symbols('A B x y')
    f = A*x + B  # linear func y=mx+b
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

def test():
    xarr,yarr = ([1,2,3,4],[2,4,6,8])
    A,B,x,y = sp.symbols('A B x y')
    f = A*x + B  # linear func y=mx+b
    e = (f - y)**2  # error squared
    sc = evalSumF(e,x,y,setupData(5))
    print (f,e)
    print ('full cost expansion: ', sc)
    print ('partial A',sp.diff(sc,sp.symbols('A')))
    print ('partial B',sp.diff(sc,sp.symbols('B')))
    print ('done')

#return in array with original result in [0], timing in [1]
def time_fn( fn, *args, **kwargs ):
    start = time.clock()
    results = fn( *args, **kwargs )
    end = time.clock()
    fn_name = fn.__module__ + "." + fn.__name__
    #print fn_name + ": " + str(end-start) + "s"
    return [results,end-start]

timings = []
dfs = makeFakeData()
for d in dfs[0:2]:
    r = time_fn(grad_descent2,d)
    print ('finished for rows,time(s)',d.shape[0], r[1])
    timings.append(r)
print('*** done')
print(timings)