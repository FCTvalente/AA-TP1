# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import StratifiedKFold
from KernelDensityNB import KernelDensityNB
from comparator import mcNemarTest, score, normalTest


def standardize(vec):
    res = 0
    for y in range(vec.shape[1]):
        if y == 0:
            res = (vec[:,y] - np.mean(vec[:,y], axis=0))/np.std(vec[:,y], axis=0)
        else:
            res = np.column_stack((res, (vec[:,y] - np.mean(vec[:,y], axis=0))/np.std(vec[:,y], axis=0)))
        
    return res
    

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
trainXs = standardize(trainXs)

testdata = np.loadtxt('TP1_test.tsv')

testYs = testdata[:,-1]
testXs = testdata[:,:-1]
testXs = standardize(testXs)


plt.figure()
bw = 0.02
bbw = 0.02
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
        bbw = bw
        BestKDE = temKDE
    bw += 0.02
    
plt.title('KDE error for bandwidth optimisation (Best bandwidth: {0:1.2f})'.format(bbw))
x = np.linspace(0.02, 0.6, len(KDE_tr_err))
plt.plot(x, KDE_tr_err, '-', label='Training error')
plt.plot(x, KDE_va_err, '-', label='Cross-validation error')
plt.plot(bbw, bva_error, 'X')
plt.legend()
#plt.savefig(nbimage,dpi=300,bbox_inches="tight")
plt.show()
plt.close()

BestKDE.fit(trainXs, trainYs)


GNB = GaussianNB()
GNB.fit(trainXs, trainYs)


plt.figure()
bgamma = 0.2
bva_error = 1
gamma = 0.2
SVM_tr_err = []
SVM_va_err = []
temSVM = 0
BestSVM = 0
while gamma <= 6:
    tr_err, va_err, temSVM = svm_cv(trainXs, trainYs, gamma, 5)
    SVM_tr_err.append(tr_err)
    SVM_va_err.append(va_err)
    if va_err < bva_error:
        BestSVM = temSVM
        bva_error = va_err
        bgamma = gamma
    gamma += 0.2

plt.title('SVC error for gamma optimisation (Best gamma: {0:1.2f})'.format(bgamma))
x = np.linspace(0.2, 6, len(SVM_tr_err))
plt.plot(x, SVM_tr_err, '-', label='Training error')
plt.plot(x, SVM_va_err, '-', label='Cross-validation error')
plt.plot(bgamma, bva_error, 'X')
plt.legend()
#plt.savefig(svmimage,dpi=300,bbox_inches="tight")
plt.show()
plt.close()

BestSVM.fit(trainXs, trainYs)

KDEpred = BestKDE.predict(testXs)
GNBpred = GNB.predict(testXs)
SVMpred = BestSVM.predict(testXs)

KDEmiss, KDEsigma = score(testYs, KDEpred)
GNBmiss, GNBsigma = score(testYs, GNBpred)
SVMmiss, SVMsigma = score(testYs, SVMpred)

KDEvsGNBmc = mcNemarTest(testYs, KDEpred, GNBpred)
GNBvsSVMmc = mcNemarTest(testYs, GNBpred, SVMpred)
SVMvsKDEmc = mcNemarTest(testYs, SVMpred, KDEpred)

KDEvsGNBnm = normalTest(KDEmiss, GNBmiss, KDEsigma, GNBsigma)
GNBvsSVMnm = normalTest(GNBmiss, SVMmiss, GNBsigma, SVMsigma)
SVMvsKDEnm = normalTest(SVMmiss, KDEmiss, SVMsigma, KDEsigma)

KDETrueError = 1 - BestKDE.score(testXs, testYs)
GNBTrueError = 1 - GNB.score(testXs, testYs)
SVMTrueError = 1 - BestSVM.score(testXs, testYs)
sform = '{0}: true error = {1}'
print(sform.format('KDE',KDETrueError))
print(sform.format('GNB',GNBTrueError))
print(sform.format('SVM',SVMTrueError))

sform = '{0}: misses = {1} ; sigma = {2}'
print(sform.format('KDE', KDEmiss, KDEsigma))
print(sform.format('GNB', GNBmiss, GNBsigma))
print(sform.format('SVM', SVMmiss, SVMsigma))

sform = '{0}: Normal Test => {1} ; McNemar Test => {2}'
print(sform.format('KDE vs GNB', KDEvsGNBnm, KDEvsGNBmc))
print(sform.format('GNB vs SVM', GNBvsSVMnm, GNBvsSVMmc))
print(sform.format('SVM vs KDE', SVMvsKDEnm, SVMvsKDEmc))