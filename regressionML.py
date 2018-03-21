import requests, pandas, io
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum

def setupData(max=1000):
    url='http://www.stat.ufl.edu/~winner/data/brainhead.dat'
    data=requests.get(url)
    col_names=('gender', 'age_range', 'head_size', 'brain_weight')
    col_widths=[(8,8),(16,16),(21-24),(29-32)]
    df=pandas.read_fwf(io.StringIO(data.text), names=col_names, colspec=col_widths)
    print (df.head(5))
    return df.head(max)

# iterator for series
def costP(func, x, testData):
    n = 0
    print('test data size: ',len(testData))
    for _,d in testData.iterrows():  # global test data
        n += (as_int(func.subs(x,d.head_size)) - d.brain_weight)**2
    return n * (1.0/len(testData))

# make function f, and error sq function e
def makeFuncs():
    A,B,x,y = sp.symbols('A B x y')
    f = A*x + B  # linear func y=mx+b
    errorF = (f - y)**2
    return f, errorF

# evaluate/calculate e with data for x and y
def sumCalcF(e,testData):
    n=0
    for _,d in testData.iterrows():  # global test data
        n += e.subs(sp.symbols('x'),d.head_size).subs(sp.symbols('y'),d.brain_weight)
    return n * (1.0/len(testData))

# generate partial derivative of e, with respect to v, for testData (x,y) and evaluate
def partialDeriv(e,testData,v,guessV,o,guessO):
    p = sp.diff(e,v)
    pc = sumCalcF(p,testData)
    pceval = pc.subs(v,guessV).subs(o,guessO)
    print ('v,p,pc:pceval',v,guessV,p,pc,pceval)
    return pceval

# solver, start w/ guess, solve cost, iterate cost+/-partialDerivs
def grad_descent2():
    guessA = 1.0
    guessB = 100.0   
    testData = setupData()
    step = 0.00000004
    step_limit = 0.0001 # when to stop, when stops changing
    changeA = changeB = 1.0  # initial guess 1,1 or y=x+1
    ccChange = 1

    A,B = sp.symbols('A B')
    f, e = makeFuncs()  # make error = (Ax+B - y)^2
    c = sumCalcF(e,testData)  # cost fun evaluted for testData
    cc = c.subs(A,guessA).subs(B,guessB)  # cost evaluted for A B guess
    print('init cost',cc)

    i=0
    #for each x in the training set:
#    while (abs(changeA) > step_limit) or (abs(changeB) > step_limit):
#    while (abs(ccChange) > step_limit):
    while(i<5000):
        pda = partialDeriv(e,testData,A,guessA,B,guessB)
        pdb = partialDeriv(e,testData,B,guessB,A,guessA)
        changeA = step * pda
        changeB = step * pdb
        guessA = guessA - changeA
        guessB = guessB - changeB
        previousCC = cc
        cc = c.subs(A,guessA).subs(B,guessB)
        ccChange = previousCC-cc
        print ('A,B,-A,-B', guessA, guessB, changeA, changeB)
        print ('     current loop/cost',i,cc)
        i=i+1
    return guessA,guessB

def test():
    xarr,yarr = ([1,2,3,4],[2,4,6,8])
    f,e = makeFuncs()
    sc = sumCalcF(e,setupData(5))
    print (f,e)
    print ('full cost expansion: ', sc)
    print ('partial A',sp.diff(sc,sp.symbols('A')))
    print ('partial B',sp.diff(sc,sp.symbols('B')))
    print ('done')

    print(grad_descent2())

test()

