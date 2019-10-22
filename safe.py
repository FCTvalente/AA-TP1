# -*- coding: utf-8 -*-

i = 0
        while i< len(preds):
            print(sum1[i])
            print(sum0[i])
            if(sum0[i] < sum1[i]):
                preds[i] = 1
                
            i += 1
            

def kde_fold(Xs, Ys, tr_ix, va_ix, bandwidth):
    tr_Xs = Xs[tr_ix]
    tr_Ys = Ys[tr_ix]
    tr_Xs1 = tr_Xs[tr_Ys == 1,:]
    tr_Xs0 = tr_Xs[tr_Ys == 0,:]
    
    va_Xs = Xs[va_ix,:]
    
    prob1 = np.log(len(tr_Xs1) / len(tr_Ys[:]))
    prob0 = np.log(len(tr_Xs0) / len(tr_Ys[:]))
    
    kdeP0 = KernelDensity(bandwidth=bandwidth)
    kdeP0.fit(tr_Xs0[:,[0]])
    arrP0 = kdeP0.score_samples(va_Xs[:,[0]])
    kdeP1 = KernelDensity(bandwidth=bandwidth)
    kdeP1.fit(tr_Xs0[:,[1]])
    arrP1 = kdeP1.score_samples(va_Xs[:,[1]])
    kdeP2 = KernelDensity(bandwidth=bandwidth)
    kdeP2.fit(tr_Xs0[:,[2]])
    arrP2 = kdeP2.score_samples(va_Xs[:,[2]])
    kdeP3 = KernelDensity(bandwidth=bandwidth)
    kdeP3.fit(tr_Xs0[:,[3]])
    arrP3 = kdeP3.score_samples(va_Xs[:,[3]])
    
    sump = [prob1 + (arrP0[i]+arrP1[i]+arrP2[i]+arrP3[i]) for i in range(len(arrP0))]
    
    kdeN0 = KernelDensity(bandwidth=bandwidth)
    kdeN0.fit(tr_Xs1[:,[0]])
    arrN0 = kdeN0.score_samples(va_Xs[:,[0]])
    kdeN1 = KernelDensity(bandwidth=bandwidth)
    kdeN1.fit(tr_Xs1[:,[1]])
    arrN1 = kdeN1.score_samples(va_Xs[:,[1]])
    kdeN2 = KernelDensity(bandwidth=bandwidth)
    kdeN2.fit(tr_Xs1[:,[2]])
    arrN2 = kdeN2.score_samples(va_Xs[:,[2]])
    kdeN3 = KernelDensity(bandwidth=bandwidth)
    kdeN3.fit(tr_Xs1[:,[3]])
    arrN3 = kdeN3.score_samples(va_Xs[:,[3]])
    
    sumn = [prob0 + (arrN0[i]+arrN1[i]+arrN2[i]+arrN3[i]) for i in range(len(arrN0))]
    
    classifier = np.zeros(len(sump))
    classifier[sumn < sump] = 1
   
    return 0, 0, classifier


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
