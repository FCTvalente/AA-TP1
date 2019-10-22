# -*- coding: utf-8 -*-

import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import accuracy_score

class KernelDensityNB:
    
    def __init__(self, bandwidth, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, Xs, Ys):
        self.classifiers1 = []
        self.classifiers0 = []
        Xs1 = Xs[Ys[:]==1,:]
        Xs0 = Xs[Ys[:]==0,:]
        for z in range(Xs.shape[1]):
            k1 = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            k1.fit(Xs1[:,[z]])
            self.classifiers1.append(k1)
            k0 = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            k0.fit(Xs0[:,[z]])
            self.classifiers0.append(k0)
        
        self.prob1 = np.log(len(Xs1) / len(Ys))
        self.prob0 = np.log(len(Xs0) / len(Ys))
        
    def score(self, Xs, Ys):
        preds1 = []
        preds0 = []
        sum1 = np.zeros(len(Xs))
        sum0 = np.zeros(len(Xs))
        for z in range(len(self.classifiers1)):
            pred1 = self.classifiers1[z].score_samples(Xs[:,[z]])
            preds1.append(pred1)
            pred0 = self.classifiers0[z].score_samples(Xs[:,[z]])
            preds0.append(pred0)
            
        for z in range(len(Xs)):
            sum1[z] = self.prob1 + preds1[0][z] + preds1[1][z] + preds1[2][z] + preds1[3][z]
            sum0[z] = self.prob0 + preds0[0][z] + preds0[1][z] + preds0[2][z] + preds0[3][z]
        
        preds = np.zeros(len(Xs))
        preds[sum0 < sum1] = 1
        res = accuracy_score(preds, Ys)
        return res
    
    