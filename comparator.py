
import math

def tableBuilt(true,first,second):
    table = [0,0]
    for i in range(0,len(true)):
        if true[i] != first[i] and true[i] == second[i]:
            table[0] += 1
        if true[i] == first[i] and true[i] != second[i]:
            table[1] += 1
    return table;


def mcNemarTest(true, first, second):
    table = tableBuilt(true,first, second)
    result = -1
    den = table[1]+table[0]
    num = (abs(table[0]-table[1])-1)**2
    result = num/den
    return result


def score(true, classifier):
    missChance = 0
    #x
    miss = 0
    #N
    NOfValues = len(true)
    for i in range(0,NOfValues):
        if(true[i]!=classifier[i]):
            miss += 1
    #p0
    missChance = miss/NOfValues
    
    sigma =  math.sqrt(NOfValues*missChance*(1-missChance))
    
    confidenceSigma = sigma* 1.96
    return miss,confidenceSigma

def normalTest(missClassification1,missClassification2,confidenceSigma1,confidenceSigma2):
    
    upper1= missClassification1 + confidenceSigma1
    upper2= missClassification2 + confidenceSigma2
    lower1= missClassification1 - confidenceSigma1
    lower2= missClassification2 - confidenceSigma2
    
    
    best = 'Collision'
    if(not(lower1 <= upper2 and lower2 <= upper1)):
        if(missClassification1 > missClassification2):
            best = 'Second'
        else:
            best = 'First'
    return best

    
    