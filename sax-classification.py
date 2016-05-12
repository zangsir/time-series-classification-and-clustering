from saxpy import SAX
from sklearn.metrics import classification_report
import matplotlib.pylab as plt
import numpy as np
import random
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import csv
import random
import math
import operator
import sys
from os import listdir
from os.path import isfile, join


def split_data(data,test_size):
    test_ind = random.sample(range(len(data)),test_size)
    whole_ind = range(len(data))
    train_ind = [x for x in whole_ind if x not in test_ind]
    test = np.array([data[x] for x in test_ind])
    train = np.array([data[x] for x in train_ind])
    return test, train

def min_dist_sax(t1String,t2String,word,alpha,eps=0.000001):
    s=SAX(word,alpha,eps)
    return s.compare_strings(t1String,t2String)


def convert_sax(ts,word,alpha,eps=0.000001):
    s=SAX(word,alpha,eps)
    (t1String, t1Indices) = s.to_letter_rep(ts)
    return t1String


#convert all data to sax
def data_sax(data,word,alpha):
    data_sax=[]
    for ts in data:
        ts_string=convert_sax(ts[:-1],word,alpha)
        ts_string+=str(int(ts[-1]))
        data_sax.append(ts_string)
    return data_sax




def knn_sax(train,test,word,alpha):
    preds=[]
    #index, line value
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        #HeRE i and j are just two strings 'abcd1' and 'abcc2' e.g.
        for j in train:
            dist=min_dist_sax(i[:-1],j[:-1],word,alpha)
            if dist<min_dist:
                min_dist=dist
                closest_seq=j
                #print 'j',j
        preds.append(closest_seq[-1])
    print 'preds',preds
    gtruth=[x[-1] for x in test]
    print gtruth
    prec = precision_score(gtruth,preds, average='macro')
    recall= recall_score(gtruth,preds, average='macro')
    f1=f1_score(gtruth,preds, average='macro')

    return (prec,recall,f1)





#convert all data to sax
def data_sax(data,word,alpha):
    data_sax=[]
    for ts in data:
        ts_string=convert_sax(ts[:-1],word,alpha)
        ts_string+=str(int(ts[-1]))
        data_sax.append(ts_string)
    return data_sax

def generate_sax_data(data):
    all_sax=[]
    all_pars=[]
    wpars=[15]
    apars=[14,17,20]
    for word in wpars:
        for alpha in apars:
            a=data_sax(data,word,alpha)
            all_sax.append(a)
            all_pars.append((word,alpha))
    return all_sax,all_pars
        





def run_1nn(all_data,num_iter=1):
    print 'running 1nn'
    score_dict_sax={}
    for data_file in all_data:
        all_sax,all_pars=generate_sax_data(data_file)
        print 'generated file from ' + str(len(all_data)) + ' data file(s)'
        for i in range(len(all_sax)):
            data=all_sax[i]
            data_pars=all_pars[i]
            total_score=(0,0,0)
            print "sax parameters:",data_pars
            #fileName=mypath+file
            #print fileName
            #data = np.genfromtxt(fileName, delimiter=',')
            for i in range(num_iter):
                best_so_far=(0,0,0)
                test,train=split_data(data,1600)
                scores = knn_sax(train,test,data_pars[0],data_pars[1])
                print 'scores:',scores
                if scores[-1]>best_so_far[-1]:
                    best_so_far=scores                
                print "scores:",best_so_far
                total_score=addv(total_score,scores)
                print "total score:",total_score
            ave_score=[x/num_iter for x in total_score]
            print ave_score
            #score_dict_sax[file]=ave_score
            print "======================================="
            
            
def getNeighborsSAX(trainingSet, testInstance, word,alpha):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = min_dist_sax(trainingSet[x][:-1],testInstance[:-1],word,alpha)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
            
            
def run_knn(all_data,k,num_iter=1):
    print 'running knn'
    score_dict_sax={}
    for data_file in all_data:
        all_sax,all_pars=generate_sax_data(data_file)
        print 'generated file from ' + str(len(all_data)) + ' data file(s)'
        for i in range(len(all_sax)):
            data=all_sax[i]
            data_pars=all_pars[i]
            total_score=(0,0,0)
            print "sax parameters:",data_pars
            #fileName=mypath+file
            #print fileName
            #data = np.genfromtxt(fileName, delimiter=',')
            for i in range(num_iter):
                predictions=[]
                test,train=split_data(data,1600)
                for x in range(len(test)):
                    neighbors = getNeighborsSAX(train, test[x], data_pars[0],data_pars[1])
                    result = getResponse(neighbors)
                    predictions.append(result)
                    #print('> predicted=' + repr(result) + ', actual=' + repr(test[x][-1]))
                accuracy = getAccuracy(test, predictions)
                print('Accuracy: ' + repr(accuracy) + '%')
                gtruth=[x[-1] for x in test]
                prec = precision_score(gtruth,predictions, average='macro')
                recall= recall_score(gtruth,predictions, average='macro')
                f1=f1_score(gtruth,predictions, average='macro')
                print prec,recall,f1

            
            


mypath="toneSub/"
allfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
onlyfiles=[f for f in allfiles if f.endswith('csv')]
all_data=[]

k=int(sys.argv[1])


for file in onlyfiles:
    total_score=(0,0,0)
    fileName=mypath+file
    print fileName
    data = np.genfromtxt(fileName, delimiter=',')
    all_data.append(data)

    
if k==1:
    run_1nn(all_data)
elif k>1:
    run_knn(all_data,k)