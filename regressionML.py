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
    print (df.head(10))
    return df.head(max)

# iterate series
def costP(func, x, testData):
    n = 0
    print('test data size: ',len(testData))
    for _,d in testData.iterrows():  # global test data
        n += (as_int(func.subs(x,d.head_size)) - d.brain_weight)**2
    return n * (1.0/len(testData))

# make function f, and error sq function
def makeFuncs():
    A,B,x,y = sp.symbols('A B x y')
    f = A*x + B  # linear func y=mx+b
    errorF = (f - y)**2
    return f, errorF

# evaluates e with x,y substitutions
def calcF(e,x,y): 
    return e.subs(sp.symbols('x'),x).subs(sp.symbols('y'),y)

# evaluate/calculate e with data for x and y
def sumCalcF(e,testData):
    n=0
    for _,d in testData.iterrows():  # global test data
#        n += calcF(e, d.head_size,d.brain_weight)
        n += e.subs(sp.symbols('x'),d.head_size).subs(sp.symbols('y'),d.brain_weight)
    return n * (1.0/len(testData))

def partialDeriv(e,testData,v,guessV,o,guessO):
    p = sp.diff(e,v)
    pc = sumCalcF(p,testData)
    pceval = pc.subs(v,guessV).subs(o,guessO)
    print ('p:pc:pceval',v,p,pc,pceval)
    return pceval

def grad_descent2():
    guessA = 0.40
    guessB = 300.0   # h(x) = Ax+B = x+1
    testData = setupData()
    step = 0.00000001
    step_limit = 0.01 # when to stop, when stops changing
    changeA = changeB = 1.0  # initial guess 1,1 or y=x+1

    A,B = sp.symbols('A B')
    f, e = makeFuncs()
    c = sumCalcF(e,testData)
    cc = c.subs(A,guessA).subs(B,guessB)
    print('init cost',cc)

    z=0
    #for each x in the training set:
    while (abs(changeA) > step_limit) or (abs(changeB) > step_limit):
        pda = partialDeriv(e,testData,A,guessA,B,guessB)
        pdb = partialDeriv(e,testData,B,guessB,A,guessA)
        print ('pda/pdb',pda,pdb)
        changeA = step * pda
        changeB = step * pdb
        guessA = guessA - changeA
        guessB = guessB - changeB
        cc = c.subs(A,guessA).subs(B,guessB)
        print ('A,B', guessA, guessB, changeA, changeB)
        print ('     iter cost',cc)
#        z=z+1
#        if (z > 20):  #stop at 20 for now
#           return guessA,guessB
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

