import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
from mpmath import *
import logging as log

# copied code from meng - pull in songclass/* lady gaga/class music text data
def getGagaData(maxrows=200,maxfeatures=4000):
    import random, sklearn, sklearn.feature_extraction.text, sklearn.naive_bayes

    def append_data(ds,dir,label,size):
        filenames=os.listdir(dir)
        for i,fn in enumerate(filenames):
            if (i>=size):
                break            
            data=open(dir+'/'+fn,'r').read()
            ds.append((data,label))
        return ds

    ######## Load the raw data
    dataset=[]
    append_data(dataset,'songclass/lyrics/gaga',1,maxrows/2)
    append_data(dataset,'songclass/lyrics/clash',0,maxrows/2)

    log.debug("gaga test set %i docs, training set %i docs" % (len(dataset),len(dataset)))

    ######## Train the algorithm with the labelled examples (training set)
    data,target=zip(*dataset)
    vec=sklearn.feature_extraction.text.CountVectorizer()
    mat=vec.fit_transform(data)
    yarr = list(target)
    data = mat.toarray()
    labels = vec.get_feature_names()[0:maxfeatures]

    # hack trim features by 'maxfeature' param
    if (maxfeatures > len(data[0])):
        maxfeatures = len(data[0]) 
    data = data[:,0:maxfeatures]

    return data,yarr,labels


# test with mike eng's dataset
def testGaga():
    trainingMatrix,yArr,labels = getGagaData(maxrows=300,maxfeatures=20)

    log.debug (trainingMatrix)
    log.debug (yArr)

    ts = sp.symbols('t:'+str(len(trainingMatrix[0])))  #theta weight/parameter array
    xs = sp.symbols('x:'+str(len(trainingMatrix[0])))  #feature array

    c,g,h,y = sp.symbols('c g h y')
    h = (sp.Matrix([ts])*sp.Matrix(xs))[0] # multipy ts's * xs's ( ts * xs.T )
    g = 1 / (1+mp.e**-h)   # wrap h in sigmoid
    c = -y*sp.log(g) - (1-y)*sp.log(1-g)  # cost func of single sample

    log.warn ('g: %s',g)
    log.warn ('c: %s',c)
    log.warn ('tMatrix: %s',trainingMatrix)
    log.warn ('yArr: %s',yArr)
    log.warn('columns: %s',labels)
    grad_descent4(g,c,trainingMatrix,yArr)

def testFeatureCleanup():
    trainingMatrix,yArr,labels = getGagaData()
    numpy.set_printoptions(linewidth=163)
    numpy.set_printoptions(threshold='nan')

    counts = {0:0}
    words = {}
    for i,col in enumerate(trainingMatrix.T):
        sum = numpy.sum(col)
        if (sum not in counts):
            counts[sum] = 1
        else:
            counts[sum] = counts[sum] + 1
        words[labels[i]+'-'+str(i)] = sum

    print (counts)
    import operator
    sorted_words = sorted(words.items(), key=operator.itemgetter(1))
    for s,t in sorted_words:
        print (s,t)

log.basicConfig(level=log.WARN)
log.info('start %s'%(log.getLogger().level))
#testLR()
#testLR2()
#testGaga()
testFeatureCleanup()
