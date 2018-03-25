import requests, pandas, io
import itertools
import time

#return in array with original result in [0], timing in [1]
def time_fn( fn, *args, **kwargs ):
    start = time.clock()
    results = fn( *args, **kwargs )
    end = time.clock()
    #fn_name = fn.__module__ + "." + fn.__name__
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

# matrix method for gradient descent
def grad_descent3(x,y):
    guessA = guessB = 1.0
    return guessA,guessB

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

testLD3()
