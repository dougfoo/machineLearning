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
    print (df.head(max))
    return df.head(max)

# iterate series
def costP(func, x, testData):
    n = 0
    print('test data size: ',len(testData))
    for _,d in testData.iterrows():  # global test data
        n += (as_int(func.subs(x,d.head_size)) - d.brain_weight)**2
    return n * (1.0/len(testData))

# non interative using series summation (can't iterate on i - dead)
def makeFuncs(xarr, yarr):
    n = len(xarr)
    A,B,x,f,costF,i,n = sp.symbols('A B x f costF i n')
    f = A*x + B  # line func
    costF = 1.0/n * (sum.summation((f.subs(x,xarr[1])-yarr[1])**2,(i,0,n-1)))  #can't input array[i]
    return f, costF

def makeFuncs2():
    A,B,x,f,costF,i,n,y = sp.symbols('A B x f costF i n y')
    f = A*x + B  # line func
    errorF = (f - y)**2
    return f, errorF

def costF(e,x,y):
    return e.subs(sp.symbols('x'),x).subs(sp.symbols('y'),y)

def sumCostF(e,testData):
    n=0
    for _,d in testData.iterrows():  # global test data
        n += costF(e, d.head_size,d.brain_weight)
    return n * (1.0/len(testData))

def derivF(c,v):
    return sp.diff(c,v)

xarr,yarr = ([1,2,3,4],[2,4,6,8])
f,e = makeFuncs2()
sc = sumCostF(e,setupData(5))
print (f,e)
print ('full cost expansion: ', sc)
print ('partial A',derivF(sc,sp.symbols('A')))
print ('partial B',derivF(sc,sp.symbols('B')))
print ('done')

def grad_descent2():
    guessA = guessB = 1   # h(x) = Ax+B = x+1
    testData = setupData(5)
    step = 0.05
    step_limit = 0.01 # when to stop
    changeA = changeB = 1

    f, c = makeFuncs([1,2,3,4],[2,4,6,8]) #replace w/ testData X|Y

    f.subs(A, guessA) # f = ?
    f.subs(B, guessB)
    print('f',f)
    print('cost ',c)
    
    #for each x in the training set:
    while (changeA < step_limit) and (changeB < step_limit):
        changeA = step * sp.diff(c,A)
        changeB = step * sp.diff(c,B)
        guessA = guessA - changeA
        guessB = guessB - changeB
        print (guessA, guessB)
    return guessA,guessB

def test():
    guessA = guessB = 1   # h(x) = Ax+B = x+1
    testData = setup()

    A,B,x,f,f2 = sp.symbols('A B x f f2')
    f = guessA*x +guessB
    print('f',f)
    z = f.subs(x,3)
    print (type(z))


    print (type(as_int(z)))
    print (z)

    c = costP(f, x, testData)
    print('init cost ',c)

    i,a,b = sp.symbols('i a b')
    a=1
    b=5
    #print (sum.summation(f*i,(i,a,b)))
    #print (sum.summation(f2*i,(i,a,b)))
    f2 = A*x*5 + B
    print (f2*i,type(f2*i))
    s = sum.summation(f2*i,(i,a,b))

    print (sp.diff(sum.summation(f2*i,(i,a,b)),A))
    print (sp.diff(sum.summation(f2*i,(i,a,b)),B))

    #sp.init_printing() # what this do

    """ 
    A,B,x,f = sp.symbols('A B x f')
    f = (A * x) + B
    print (f)
    x = A + 3
    sp.solve(x,1)   # how to solve for x = A+3 where A=1?  x = 4

    n=5
    print (x.evalf(subs={A: 2}))
    print (f.evalf(subs={B: 3}))
    print (x.subs(A,3))
    print (f.subs(B,4))


    # how to do f(A=3, B=1) = 3x + 1 ?

    print ('diff to A: ', sp.diff(A*x+B, A))
    print ('diff to B: ', sp.diff(A*x+B, B))
    print ('f=',f)


    """