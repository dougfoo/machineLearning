import requests, pandas, io
import itertools
import time
import myutils

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
