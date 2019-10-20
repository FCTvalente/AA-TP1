# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import StratifiedKFold

def kde_fold2(tr_Xs, tr_Ys, va_Xs, va_Ys, prob0, prob1, bandwidth):
    kdeP = KernelDensity(bandwidth=bandwidth)
    kdeP.fit(tr_Xs[tr_Ys[:]==1,:])
    arrP = kdeP.score_samples(va_Xs[va_Ys[:]==1,:])
    sump = [np.log(prob1) + arrP[i] for i in range(len(arrP))]
    
    kdeN = KernelDensity(bandwidth=bandwidth)
    kdeN.fit(tr_Xs[tr_Ys[:]==0,:])
    arrN = kdeN.score_samples(va_Xs[va_Ys[:]==0,:])
    sumn = [np.log(prob0) + arrN[i] for i in range(len(arrN))]
    
    classifier = [max(sump[i], sumn[i]) for i in range(len(sump))]
    return classifier

def kde_fold(tr_Xs, tr_Ys, va_Xs, va_Ys, prob0, prob1, bandwidth):
    kdeP0 = KernelDensity(bandwidth=bandwidth)
    kdeP0.fit(tr_Xs[tr_Ys[:]==1,0])
    arrP0 = kdeP0.score_samples(va_Xs[va_Ys[:]==1,0])
    kdeP1 = KernelDensity(bandwidth=bandwidth)
    kdeP1.fit(tr_Xs[tr_Ys[:]==1,1])
    arrP1 = kdeP1.score_samples(va_Xs[va_Ys[:]==1,1])
    kdeP2 = KernelDensity(bandwidth=bandwidth)
    kdeP2.fit(tr_Xs[tr_Ys[:]==1,2])
    arrP2 = kdeP2.score_samples(va_Xs[va_Ys[:]==1,2])
    kdeP3 = KernelDensity(bandwidth=bandwidth)
    kdeP3.fit(tr_Xs[tr_Ys[:]==1,3])
    arrP3 = kdeP3.score_samples(va_Xs[va_Ys[:]==1,3])
    
    sump = [np.log(prob1) + np.log(arrP0[i]+arrP1[i]+arrP2[i]+arrP3[i]) for i in range(len(arrP0))]
    
    kdeN0 = KernelDensity(bandwidth=bandwidth)
    kdeN0.fit(tr_Xs[tr_Ys[:]==0,0])
    arrN0 = kdeN0.score_samples(va_Xs[va_Ys[:]==0,0])
    kdeN1 = KernelDensity(bandwidth=bandwidth)
    kdeN1.fit(tr_Xs[tr_Ys[:]==0,1])
    arrN1 = kdeN1.score_samples(va_Xs[va_Ys[:]==0,1])
    kdeN2 = KernelDensity(bandwidth=bandwidth)
    kdeN2.fit(tr_Xs[tr_Ys[:]==0,2])
    arrN2 = kdeN2.score_samples(va_Xs[va_Ys[:]==0,2])
    kdeN3 = KernelDensity(bandwidth=bandwidth)
    kdeN3.fit(tr_Xs[tr_Ys[:]==0,3])
    arrN3 = kdeN3.score_samples(va_Xs[va_Ys[:]==0,3])
    
    sumn = [np.log(prob0) + np.log(arrN0[i]+arrN1[i]+arrN2[i]+arrN3[i]) for i in range(len(arrN0))]
    
    classifier = [max(sump[i], sumn[i]) for i in range(len(sump))]
    return classifier


def kde_cv(Xs, Ys, bandwidth, folds):
    classifier = []
    kf = StratifiedKFold(n_splits = folds)
    tr_err = 0
    va_err = 0
    for tr_ix, va_ix in kf.split(Ys, Ys):
        tr_Xs = Xs[tr_ix]
        tr_Ys = Ys[tr_ix]
        va_Xs = Xs[va_ix]
        va_Ys = Ys[va_ix]
        
        psum = len(tr_Ys[tr_Ys[:]==1])
        nsum = len(tr_Ys[tr_Ys[:]==0])
        prob0 = nsum / len(tr_Ys[:])
        prob1 = psum / len(tr_Ys[:])
        classfold = kde_fold2(tr_Xs, tr_Ys, va_Xs, va_Ys, prob0, prob1, bandwidth)
        #plt.plot(classfold)
        tr_err += -1
        va_err += -1
    return tr_err/folds, va_err/folds

def svm_cv(Xs, Ys, gamma, folds, opt=1):
    kf = StratifiedKFold(n_splits = folds)
    tr_err = 0
    va_err = 0
    for tr_ix, va_ix in kf.split(Ys, Ys):
        mach = svm.SVC(C = opt, kernel = 'rbf', gamma = gamma, probability=True)
        mach.fit(Xs[tr_ix,:],Ys[tr_ix])
        prob = mach.predict_proba(Xs[:,:])[:,1]
        squares = (prob-Ys)**2
        tr_tem, va_tem = np.mean(squares[tr_ix]),np.mean(squares[va_ix])
        tr_err += tr_tem
        va_err += va_tem
    return tr_err, va_err


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


plt.figure()
plt.title('PLACEHOLDER')
bw = 0.02
KDE_tr_err = []
KDE_va_err = []
while bw <= 0.6:
    tr_err, va_err = kde_cv(trainXs, trainYs, bw, 5)
    KDE_tr_err.append(tr_err)
    KDE_va_err.append(va_err)
    bw += 0.02
    
x = np.linspace(0.02, 0.6, len(KDE_tr_err))
plt.plot(x, KDE_tr_err)
plt.plot(x, KDE_va_err)
plt.show()
#plt.savefig(nbimage,dpi=300,bbox_inches="tight")
plt.close()


GNB = GaussianNB()
GNB.fit(trainXs, trainYs)


plt.figure()
plt.title('PLACEHOLDER')
gamma = 0.2
SVM_tr_err = []
SVM_va_err = []
while gamma <= 6:
    tr_err, va_err = svm_cv(trainXs, trainYs, gamma, 5)
    SVM_tr_err.append(tr_err)
    SVM_va_err.append(va_err)
    gamma += 0.2

x = np.linspace(0.2, 6, len(SVM_tr_err))
plt.plot(x, SVM_tr_err)
plt.plot(x, SVM_va_err)
plt.show()
#plt.savefig(svmimage,dpi=300,bbox_inches="tight")
plt.close()