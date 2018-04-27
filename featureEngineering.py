import requests, pandas, io, numpy, argparse, math
import matplotlib.pyplot as plt
from myutils import *

# copied code from meng - pull in songclass/* lady gaga/class music text data, returns (training[][],yarr[],labels[],fnames[])
def getGagaData(maxrows=200,maxfeatures=4000,gtype=None):
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
        words.append([i,labels[i],sum,cnt])
    return words, counts

# returns 3 items, [index,count,fcount] countMap{}
def countWords(trainingMatrix):
    counts = {0:0}
    words = []
    for i,col in enumerate(trainingMatrix.T):   # transpose to inspect word by word
        sum = numpy.sum(col)
        cnt = numpy.count_nonzero(col)
        if (sum not in counts):
            counts[sum] = 1
        else:
            counts[sum] = counts[sum] + 1
        words.append([i,sum,cnt])
    return words, counts

# take in 2 list[][] and merge on 1st column, fill Nan's with 0, add deltas, return DataFrame
def mergeCounts(m1,m2):
    #combine m1,m2 outer join
    g1 = pandas.DataFrame.from_records(m1,columns=['i','word','gct','gfct'])
    g2 = pandas.DataFrame.from_records(m2,columns=['i','word','nct','nfct'])

    cols = list(set(g1.columns).intersection(g2.columns))
    results = pandas.merge(g1, g2, how='outer', left_on=cols, right_on=cols)
    results = results.fillna(0)
    results['gct-delta'] = results['nct'] - results['gct']
    results['gfct-delta'] = results['nfct'] - results['gfct']
    return results

# test feature analysis/cleanup
def testScikitFeatureCleanup():
    trainingMatrix1,yArr,labels,fnames = getGagaData(maxrows=200)
    t1 = numpy.array(trainingMatrix1)

    # VarianceThreadhold
    print (t1.shape)
    from sklearn.feature_selection import VarianceThreshold
    model = VarianceThreshold(threshold=(.8 * (1 - .8)))
    m = model.fit_transform(t1)
    print ('Variance .8',m.shape)
    picklist = model.get_support(True)
    pickwords = [labels[p] for p in picklist]
    print (pickwords)

    # SelectKBest
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    X, y = t1, yArr
    print(X.shape)
    model = SelectKBest(chi2, k=50)
    X_new = model.fit_transform(X, y)   # need to keep labels
    print('KBest',X_new.shape)
    df = pandas.DataFrame(X_new)
    print (df.head())
    picklist = model.get_support(True)
    pickwords = [labels[p] for p in picklist]
    print (pickwords)

    return m

def testFeatureAnalysis():
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

    return m

log.basicConfig(level=log.WARN)
log.warn('start %s'%(log.getLogger().level))
numpy.set_printoptions(linewidth=163)
numpy.set_printoptions(threshold='nan')
#mF = testFeatureAnalysis()
#sF = testScikitFeatureCleanup()

