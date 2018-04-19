import numpy
import sklearn
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import os
import sys
import random

TESTSET_PCT=0.33   # how much of our input will we reserve for test

def append_data(ds,dir,label):
    filenames=os.listdir(dir)
    for fn in filenames:
        data=open(dir+'/'+fn,'r').read()
        ds.append((data,label))
    
    return ds

######## Load the raw data

dataset=[]
append_data(dataset,'lyrics/gaga','gaga')
append_data(dataset,'lyrics/clash','clash')

######## Partition the data into training set and test set

random.shuffle(dataset)
partition=int(TESTSET_PCT*len(dataset))
testset=dataset[:partition]
dataset=dataset[partition:]

print("test set %i docs, training set %i docs" % (len(testset),len(dataset)))

######## Train the algorithm with the labelled examples (training set)

data,target=zip(*dataset)
vec=sklearn.feature_extraction.text.CountVectorizer()
mat=vec.fit_transform(data)
classifier=sklearn.naive_bayes.MultinomialNB().fit(mat,target)

print("matrix shape" + str(mat.shape))
print("matrix\n" + str(mat))

######## Label the test set using the classifier, and measure performance

testdata,testlabel=zip(*testset)
test_mat=vec.transform(testdata)
print("prediction accuracy: ",
      numpy.mean(classifier.predict(test_mat) == testlabel)*100)

######## Examine the output

while len(testset)>1:
    testdata,testlabel=testset.pop()
    test_mat=vec.transform([testdata])
    print(testdata)

    sys.stdin.readline()

    print("Computer thinks.... ", classifier.predict(test_mat))
    sys.stdin.readline()
    print("The correct answer: ",testlabel)
    sys.stdin.readline()

    print("-------------------------------------------------------------------------------")
