# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import StratifiedKFold
from KernelDensityNB import KernelDensityNB


def McNemarSTatistic():
    return 0
    

def kde_cv(Xs, Ys, bandwidth, folds):
    kf = StratifiedKFold(n_splits = folds)
    tr_err = 0
    va_err = 0
    for tr_ix, va_ix in kf.split(Ys, Ys):
        mach = KernelDensityNB(bandwidth)
        mach.fit(Xs[tr_ix,:],Ys[tr_ix])
        tr_err += 1 - mach.score(Xs[tr_ix], Ys[tr_ix])
        va_err += 1 - mach.score(Xs[va_ix], Ys[va_ix])
    return tr_err/folds, va_err/folds, mach


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
bbw = 0.02
bva_error = 1
KDE_tr_err = []
KDE_va_err = []
BestKDE = 0
while bw <= 0.6:
    print(bw)
    tr_err, va_err, temKDE = kde_cv(trainXs, trainYs, bw, 5)
    KDE_tr_err.append(tr_err)
    KDE_va_err.append(va_err)
    if va_err < bva_error:
        bva_error = va_err
        bbw = bw
        BestKDE = temKDE
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
bgamma = 0.2
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