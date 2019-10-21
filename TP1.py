# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def kde_fold2(tr_Xs, tr_Ys, va_Xs, va_Ys, prob0, prob1, bandwidth):
    kdeP = KernelDensity(bandwidth=bandwidth)
    kdeP.fit(tr_Xs[tr_Ys[:]==1,:])
    arrP = kdeP.score_samples(va_Xs[:,:])
    sump = [np.log(prob1) + arrP[i] for i in range(len(arrP))]
    
    kdeN = KernelDensity(bandwidth=bandwidth)
    kdeN.fit(tr_Xs[tr_Ys[:]==0,:])
    arrN = kdeN.score_samples(va_Xs[:,:])
    sumn = [np.log(prob0) + arrN[i] for i in range(len(arrN))]
    
    classifier = []
    i= 0
    while i < len(sumn):
        if(sumn[i]>sump[i]):
            classifier.append(0)
        else:
            classifier.append(1)
        i += 1
    
    tr_err = 1 - np.mean(kdeP.score(tr_Xs) + kdeN.score(tr_Xs))
    va_err = 1 - np.mean(kdeP.score(va_Xs) + kdeN.score(va_Xs))
    
    return tr_err, va_err, classifier

def McNemarSTatistic():
    return 0
    
def kde_fold(tr_Xs, tr_Ys, va_Xs, va_Ys, prob0, prob1, bandwidth):
    kdeP0 = KernelDensity(bandwidth=bandwidth)
    kdeP0.fit(tr_Xs[tr_Ys[:]==1,[0]])
    arrP0 = kdeP0.score_samples(va_Xs[:,0])
    kdeP1 = KernelDensity(bandwidth=bandwidth)
    kdeP1.fit(tr_Xs[tr_Ys[:]==1,1])
    arrP1 = kdeP1.score_samples(va_Xs[:,1])
    kdeP2 = KernelDensity(bandwidth=bandwidth)
    kdeP2.fit(tr_Xs[tr_Ys[:]==1,2])
    arrP2 = kdeP2.score_samples(va_Xs[:,2])
    kdeP3 = KernelDensity(bandwidth=bandwidth)
    kdeP3.fit(tr_Xs[tr_Ys[:]==1,3])
    arrP3 = kdeP3.score_samples(va_Xs[:,3])
    
    sump = [np.log(prob1) + (arrP0[i]+arrP1[i]+arrP2[i]+arrP3[i]) for i in range(len(arrP0))]
    
    kdeN0 = KernelDensity(bandwidth=bandwidth)
    kdeN0.fit(tr_Xs[tr_Ys[:]==0,0])
    arrN0 = kdeN0.score_samples(va_Xs[:,0])
    kdeN1 = KernelDensity(bandwidth=bandwidth)
    kdeN1.fit(tr_Xs[tr_Ys[:]==0,1])
    arrN1 = kdeN1.score_samples(va_Xs[:,1])
    kdeN2 = KernelDensity(bandwidth=bandwidth)
    kdeN2.fit(tr_Xs[tr_Ys[:]==0,2])
    arrN2 = kdeN2.score_samples(va_Xs[:,2])
    kdeN3 = KernelDensity(bandwidth=bandwidth)
    kdeN3.fit(tr_Xs[tr_Ys[:]==0,3])
    arrN3 = kdeN3.score_samples(va_Xs[:,3])
    
    sumn = [np.log(prob0) + (arrN0[i]+arrN1[i]+arrN2[i]+arrN3[i]) for i in range(len(arrN0))]
    
    classifier = []
    i= 0
    while i < len(sumn):
        if(sumn[i]>sump[i]):
            classifier.append(0)
        else:
            classifier.append(1)
        i += 1
    
    return classifier


def kde_cv(Xs, Ys, bandwidth, folds):
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
        a, b, classfold = kde_fold(tr_Xs, tr_Ys, va_Xs, va_Ys, prob0, prob1, bandwidth)
        tr_err += a
        va_err += b
    return tr_err/folds, va_err/folds, classfold

def svm_cv(Xs, Ys, gamma, folds, opt=1):
    kf = StratifiedKFold(n_splits = folds)
    tr_err = 0
    va_err = 0
    for tr_ix, va_ix in kf.split(Ys, Ys):
        mach = svm.SVC(C = opt, kernel = 'rbf', gamma = gamma)
        mach.fit(Xs[tr_ix,:],Ys[tr_ix])
        tr_err += 1 - mach.score(Xs[tr_ix], Ys[tr_ix])
        va_err += 1 - mach.score(Xs[va_ix], Ys[va_ix])
    return tr_err/folds, va_err/folds, mach


traindata = np.loadtxt('TP1_train.tsv')
np.random.shuffle(traindata)

svmimage = 'SVM'
nbimage = 'NB'

trainYs = traindata[:,-1]
trainXs = traindata[:,:-1]
trainXs = (trainXs - np.mean(trainXs, axis=0))/np.std(trainXs, axis=0)

testdata = np.loadtxt('TP1_test.tsv')
#shuffle(testdata)

testYs = testdata[:,-1]
testXs = testdata[:,:-1]
testXs = (testXs - np.mean(testXs, axis=0))/np.std(testXs, axis=0)


plt.figure()
plt.title('KDE error for bandwidth optimisation')
bw = 0.02
bbw = 0
bva_error = 1
KDE_tr_err = []
KDE_va_err = []
BestKDE = 0
while bw <= 0.6:
    tr_err, va_err, temKDE = kde_cv(trainXs, trainYs, bw, 5)
    KDE_tr_err.append(tr_err)
    KDE_va_err.append(va_err)
    if va_err < bva_error:
        bva_error = va_err
        bbw = bbw
        BestKDE = -1
    bw += 0.02
    
x = np.linspace(0.02, 0.6, len(KDE_tr_err))
plt.plot(x, KDE_tr_err, 'r-', label='Training error')
plt.plot(x, KDE_va_err, 'b-', label='Cross-validation error')
plt.legend()
plt.show()
#plt.savefig(nbimage,dpi=300,bbox_inches="tight")
plt.close()


GNB = GaussianNB()
GNB.fit(trainXs, trainYs)


plt.figure()
plt.title('SVC error for gamma optimisation')
bgamma = 0
bva_error = 1
gamma = 0.2
SVM_tr_err = []
SVM_va_err = []
temSVM = 0
BestSvm = 0
while gamma <= 6:
    tr_err, va_err, temSVM = svm_cv(trainXs, trainYs, gamma, 5)
    SVM_tr_err.append(tr_err)
    SVM_va_err.append(va_err)
    if va_err < bva_error:
        BestSvm = temSVM
        bva_error = va_err
        bgamma = gamma
    gamma += 0.2

x = np.linspace(0.2, 6, len(SVM_tr_err))
plt.plot(x, SVM_tr_err, 'r-', label='Training error')
plt.plot(x, SVM_va_err, 'b-', label='Cross-validation error')
plt.legend()
plt.show()
#plt.savefig(svmimage,dpi=300,bbox_inches="tight")
plt.close()