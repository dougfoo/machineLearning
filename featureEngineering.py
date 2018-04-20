import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
import sympy as sp
from sympy.core.compatibility import as_int
import sympy.concrete.summations as sum
from myutils import *
from sklearn.utils import shuffle
import logging as log

# copied code from meng - pull in songclass/* lady gaga/class music text data
def getGagaData(maxrows=200,maxfeatures=4000,gtype=None):
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
    if (gtype == 1 or gtype == None):
        append_data(dataset,'songclass/lyrics/gaga',1,maxrows/2)
    if (gtype == 0 or gtype == None):     
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

# test feature analysis/cleanup
def testFeatureCleanup():
    numpy.set_printoptions(linewidth=163)
    numpy.set_printoptions(threshold='nan')

    trainingMatrix1,yArr1,labels1 = getGagaData(gtype=0)
    trainingMatrix2,yArr2,labels2 = getGagaData(gtype=1)

    def countWords(trainingMatrix, labels):
        counts = {0:0}
        words = {}
        for i,col in enumerate(trainingMatrix.T):   # transpose to inspect word by word
            sum = numpy.sum(col)
            if (sum not in counts):
                counts[sum] = 1
            else:
                counts[sum] = counts[sum] + 1
            words[labels[i]+'  ['+str(i)+']'] = sum

        print (i, counts)
        import operator
        sorted_words = sorted(words.items(), key=operator.itemgetter(1))
        m=numpy.asmatrix(sorted_words)      
        return m

    mGaga = countWords(trainingMatrix1, labels1)     
    mNotGaga =countWords(trainingMatrix2, labels2)


#    m = m.reshape(-1,4)
#    for a in m:
#        print ('%-25s ct: %-10s %-25s ct:%-10s'%(a.item(0),a.item(1),a.item(2),a.item(3)))


log.basicConfig(level=log.WARN)
log.info('start %s'%(log.getLogger().level))
#testLR()
#testLR2()
#testGaga()
testFeatureCleanup()
 
