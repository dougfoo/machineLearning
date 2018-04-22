import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
from myutils import *

# copied code from meng - pull in songclass/* lady gaga/class music text data
def getGagaData(maxrows=200,maxfeatures=4000,gtype=None):
    import random, sklearn, sklearn.feature_extraction.text, sklearn.naive_bayes
    def append_data(ds,dir,label,size):
        filenames=os.listdir(dir)
        for i,fn in enumerate(filenames):
            if (i>=size):
                break            
            data=open(dir+'/'+fn,'r').read()
            ds.append((data,label,fn))
        return ds### Lady Gaga 

    ######## Load the raw data
    dataset=[]
    if (gtype == 1 or gtype == None):
        append_data(dataset,'songclass/lyrics/gaga',1,maxrows/2)
    if (gtype == 0 or gtype == None):     
        append_data(dataset,'songclass/lyrics/clash',0,maxrows/2)
    log.debug("gaga test set %i docs, training set %i docs" % (len(dataset),len(dataset)))
    ######## Train the algorithm with the labelled examples (training set)
    data,target,fnames=zip(*dataset)
    vec=sklearn.feature_extraction.text.CountVectorizer()
    mat=vec.fit_transform(data)
    yarr = list(target)
    data = mat.toarray()
    labels = vec.get_feature_names()[0:maxfeatures]
    # hack trim features by 'maxfeature' param
    if (maxfeatures > len(data[0])):
        maxfeatures = len(data[0]) 
    data = data[:,0:maxfeatures]
    return data,yarr,labels,fnames

def countWords(trainingMatrix, labels, fnames):
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
    return numpy.asmatrix(sorted_words)      

# returns 2 items, [word,ct,file-ct],[#times:count]
def countWords2(trainingMatrix, labels, fnames):
    counts = {0:0}
    words = []
    for i,col in enumerate(trainingMatrix.T):   # transpose to inspect word by word
        sum = numpy.sum(col)
        cnt = numpy.count_nonzero(col)
        if (sum not in counts):
            counts[sum] = 1
        else:
            counts[sum] = counts[sum] + 1
        words.append([labels[i],sum,cnt])
    return words, counts

# take in 2 list[][] and merge on 1st column, fill Nan's with 0, add deltas, return DataFrame
def mergeCounts(m1,m2):
    #combine m1,m2 outer join
    g1 = pandas.DataFrame.from_records(m1,columns=['word','gct','gfct'])
    g2 = pandas.DataFrame.from_records(m2,columns=['word','nct','nfct'])

    cols = list(set(g1.columns).intersection(g2.columns))
    results = pandas.merge(g1, g2, how='outer', left_on=cols, right_on=cols)
    results = results.fillna(0)
    results['gct-delta'] = results['gct'] - results['nct']
    results['gfct-delta'] = results['gfct'] - results['nfct']
    return results

# test feature analysis/cleanup
def testFeatureCleanup():
    numpy.set_printoptions(linewidth=163)
    numpy.set_printoptions(threshold='nan')

    return 


#    m = m.reshape(-1,4)
#    for a in m:
#        print ('%-25s ct: %-10s %-25s ct:%-10s'%(a.item(0),a.item(1),a.item(2),a.item(3)))


log.basicConfig(level=log.WARN)
log.info('start %s'%(log.getLogger().level))
#testLR()
#testLR2()
#testGaga()
testFeatureCleanup()

numpy.set_printoptions(linewidth=163)
numpy.set_printoptions(threshold='nan')

trainingMatrix1,yArr1,labels1,fnames1 = getGagaData(gtype=0)
trainingMatrix2,yArr2,labels2,fnames2 = getGagaData(gtype=1)

m1,c1 = countWords2(trainingMatrix1, labels1, fnames1)
m2,c2 = countWords2(trainingMatrix2, labels2, fnames2)

print ('word,t-ct,f-ct',m1)
print ('counts:',c1)

m = mergeCounts(m1,m2)

print (m.shape)
print (pandas.concat([m.head(),m.tail()]))

m = m.sort_values('gct-delta')
print ('top/bot # variance of # words')
print (pandas.concat([m.head(),m.tail()]))

print ('top/bot # variance of # words (once per file)')
m = m.sort_values('gfct-delta')
print (pandas.concat([m.head(),m.tail()]))
