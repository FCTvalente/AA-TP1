# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors.kde import KernelDensity

def kde_class(Xs, Ys, bandwidth, t_Xs, t_Ys, prob0, prob1):
    kdeP0 = KernelDensity(bandwidth=bandwidth)
    kdeP0.fit(Xs[Ys[:]==1,0])
    arrP0 = kdeP0.score_samples(t_Xs[t_Ys[:]==1,0])
    kdeP1 = KernelDensity(bandwidth=bandwidth)
    kdeP1.fit(Xs[Ys[:]==1,1])
    arrP1 = kdeP1.score_samples(t_Xs[t_Ys[:]==1,1])
    kdeP2 = KernelDensity(bandwidth=bandwidth)
    kdeP2.fit(Xs[Ys[:]==1,2])
    arrP2 = kdeP2.score_samples(t_Xs[t_Ys[:]==1,2])
    kdeP3 = KernelDensity(bandwidth=bandwidth)
    kdeP3.fit(Xs[Ys[:]==1,3])
    arrP3 = kdeP3.score_samples(t_Xs[t_Ys[:]==1,3])
    
    sump = [prob1 + (arrP0[i]+arrP1[i]+arrP2[i]+arrP3[i]) for i in range(len(arrP0))]
    
    kdeN0 = KernelDensity(bandwidth=bandwidth)
    kdeN0.fit(Xs[Ys[:]==0,0])
    arrN0 = kdeN0.score_samples(t_Xs[t_Ys[:]==0,0])
    kdeN1 = KernelDensity(bandwidth=bandwidth)
    kdeN1.fit(Xs[Ys[:]==0,1])
    arrN1 = kdeN1.score_samples(t_Xs[t_Ys[:]==0,1])
    kdeN2 = KernelDensity(bandwidth=bandwidth)
    kdeN2.fit(Xs[Ys[:]==0,2])
    arrN2 = kdeN2.score_samples(t_Xs[t_Ys[:]==0,2])
    kdeN3 = KernelDensity(bandwidth=bandwidth)
    kdeN3.fit(Xs[Ys[:]==0,3])
    arrN3 = kdeN3.score_samples(t_Xs[t_Ys[:]==0,3])
    
    sumn = [prob0 + (arrN0[i]+arrN1[i]+arrN2[i]+arrN3[i]) for i in range(len(arrN0))]
    
    classifier = [max(sump[i], sumn[i]) for i in range(len(sump))]
    return classifier
    
    

traindata = np.loadtxt('TP1_train.tsv')
shuffle(traindata)

svmimage = 'SVM'
nbimage = 'NB'

trainYs = traindata[:,-1]
trainXs = traindata[:,:-1]
trainXs = (trainXs - np.mean(trainXs, axis=0))/np.std(trainXs, axis=0)

testdata = np.loadtxt('TP1_test.tsv')
shuffle(testdata)

testYs = testdata[:,-1]
testXs = testdata[:,:-1]
testXs = (testXs - np.mean(testXs, axis=0))/np.std(testXs, axis=0)

psum = len(trainYs[trainYs[:]==1,:])
nsum = len(trainYs[trainYs[:]==0,:])

prob0 = nsum / len(trainYs)
prob1 = psum / len(trainYs)

GNB = GaussianNB()
GNB.fit(trainXs, trainYs)

SupportVector = svm.SVC(C = 1, kernel = 'rbf', gamma = 0.2)
SupportVector.fit(trainXs, trainYs)