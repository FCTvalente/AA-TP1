# -*- coding: utf-8 -*-

import numpy as np
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors.kde import KernelDensity

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

SupportVector = svm.SVC(C = 1, kernel = 'rbf', gamma = 0.2)

SupportVector.fit(trainXs, trainYs)