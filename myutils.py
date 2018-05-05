import time, itertools, os,requests, pandas, io
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,log_loss
import logging as log
import sympy as sp
import numpy as np

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

#guess array formatter to 4d%f
def gf(guesses):
    if (len(guesses)>20):
        return ["{:0.4f}".format(float(g)) for g in guesses[:20]]
    else:
        return ["{:0.4f}".format(float(g)) for g in guesses]

def setupBrainData(max=1000):
    if (os.path.isfile("myDataFrame.csv")):
        log.warn('reading cache copy from disk')
        return pandas.read_csv('myDataFrame.csv').head(max)    
    url='http://www.stat.ufl.edu/~winner/data/brainhead.dat'
    data=requests.get(url)
    col_names=('gender', 'age_range', 'head_size', 'brain_weight')
    col_widths=[(8,8),(16,16),(21-24),(29-32)]
    df=pandas.read_fwf(io.StringIO(data.text), names=col_names, colspec=col_widths)
    df.to_csv('myDataFrame.csv')
    return df.head(max)

# copied code from meng - pull in songclass/* lady gaga/class music text data, returns (training[][],yarr[],labels[],fnames[])
def getGagaData(maxrows=200,maxfeatures=4000,gtype=None,stopwords=None):
    import random, sklearn, sklearn.feature_extraction.text, sklearn.naive_bayes
    def append_data(ds,dir,label,size):
        filenames=os.listdir(dir)
        for i,fn in enumerate(filenames):
            if (i>=size):
                break            
            data=open(dir+'/'+fn,'r').read()
            ds.append((data,label,fn))
        return i
    ######## Load the raw data
    dataset=[]
    if (gtype == 1 or gtype == None):
        i = append_data(dataset,'songclass/lyrics/gaga',1,maxrows/2)
        log.warn("gaga test set %i docs"%(i))
    if (gtype == 0 or gtype == None):     
        i = append_data(dataset,'songclass/lyrics/clash',0,maxrows/2)
        log.warn("non gaga test set %i docs"%(i))

    ######## Train the algorithm with the labelled examples (training set)
    data,target,fnames=zip(*dataset)
    vec=sklearn.feature_extraction.text.CountVectorizer(stop_words=stopwords)
    mat=vec.fit_transform(data)
    yarr = list(target)
    data = mat.toarray()
    labels = vec.get_feature_names()[0:maxfeatures]
    # hack trim features by 'maxfeature' param
    if (maxfeatures > len(data[0])):
        maxfeatures = len(data[0]) 
    data = data[:,0:maxfeatures]
    return data,yarr,labels,fnames

#replicate/grow data
def makeFakeData():
    print('setup expanded datasets (dfs[])')
    df = setupBrainData()
    dfs = [df,churn(df,4),churn(df,8),churn(df,12),churn(df,16)]        
    for d in dfs:
        print (d.shape)
    return dfs

# x,y is string column from dataFrame to plot on x,y axes
def plotScatter(initData,xLabel,yLabel):
    fig = plt.figure(figsize=(4,4))
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
