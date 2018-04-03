import time, itertools
import requests, pandas, io
import matplotlib.pyplot as plt

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

def setupData(max=1000):
    url='http://www.stat.ufl.edu/~winner/data/brainhead.dat'
    data=requests.get(url)
    col_names=('gender', 'age_range', 'head_size', 'brain_weight')
    col_widths=[(8,8),(16,16),(21-24),(29-32)]
    df=pandas.read_fwf(io.StringIO(data.text), names=col_names, colspec=col_widths)
    return df.head(max)

#replicate/grow data
def makeFakeData():
    print('setup expanded datasets (dfs[])')
    df = setupData()
    dfs = [df,churn(df,4),churn(df,8),churn(df,12),churn(df,16)]        
    for d in dfs:
        print (d.shape)
    return dfs

# x,y is string column from dataFrame to plot on x,y axes
def plotScatter(initData,xLabel,yLabel):
    fig = plt.figure()
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

