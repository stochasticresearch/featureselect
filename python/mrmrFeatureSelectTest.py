#!/usr/bin/env python

from sys import platform
import os

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

#miEstimators = ['ktau','knn_1','knn_6','knn_20','vme','ap','cim']
miEstimators = ['ktau','knn_1','knn_6','knn_20','ap','cim']

numCV = 10
SEED = 123
MAX_NUM_FEATURES = 20
MAX_ITER = 1000

def readArrhythmiaData():
    if platform == "linux" or platform == "linux2":
        folder = '/home/kiran/ownCloud/PhD/sim_results/arrhythmia/'
    elif platform == "darwin":
        folder = '/Users/Kiran/ownCloud/PhD/sim_results/arrhythmia/'
    elif platform == "win32":
        folder = 'C:\\Users\\kiran\\ownCloud\\PhD\\sim_results\\arrhythmia'
    z = sio.loadmat(os.path.join(folder,'X.mat'))
    X = z['X']
    y = z['y']

    miFeatureSelections = {}
    for miEstimator in miEstimators:
        featureVec = sio.loadmat(os.path.join(folder,'arrhythmia_fs_'+miEstimator+'.mat'))
        miFeatureSelections[miEstimator] = featureVec['featureVec']
    
    return (X,y,miFeatureSelections)

def evaluateClassificationPerformance(classifierStr):
    (X,y,miFeatureSelections) = readArrhythmiaData()
    y = np.squeeze(np.asarray(y))

    resultsMean = np.zeros((len(miEstimators),MAX_NUM_FEATURES))
    resultsVar = np.zeros((len(miEstimators),MAX_NUM_FEATURES))
    eIdx = 0
    for estimator in miEstimators:
        print('*'*10 + ' ' + estimator + ' ' + '*'*10)
        featureVec = np.squeeze(np.asarray(miFeatureSelections[estimator]))
        K = min(len(featureVec),MAX_NUM_FEATURES)
        for ii in range(1,K+1):
            colSelect = featureVec[0:ii]-1  # minus one to switch from index-by-1 (Matlab) 
                                            # to index-by-0 (Python)
            X_in = X[:,colSelect]
            # normalize
            X_in = preprocessing.scale(X_in)

            # we do it this way to force the random seed to be the same for each iteration
            # to compare results more consistently
            if(classifierStr=='SVC'):
                classifier = svm.SVC(random_state=SEED,max_iter=MAX_ITER)
            elif(classifierStr=='RandomForest'):
                classifier = RandomForestClassifier(random_state=SEED)
            elif(classifierStr=='KNN'):
                classifier = KNeighborsClassifier()
            elif(classifierStr=='AdaBoost'):
                classifier = AdaBoostClassifier(random_state=SEED)

            scores = cross_val_score(classifier, X_in, y, cv=numCV, n_jobs=-2)

            mu = scores.mean()
            sigma_sq = scores.std()*2
            print("numFeatures=%d Accuracy: %0.2f (+/- %0.2f)" % (ii, mu, sigma_sq))
            
            resultsMean[eIdx,ii-1] = mu
            resultsVar[eIdx,ii-1] = sigma_sq

        eIdx = eIdx + 1

    return (resultsMean,resultsVar)


if __name__=='__main__':

    classifiersToTest = ['SVC','RandomForest','KNN','AdaBoost']

    jj = 1
    for classifierStr in classifiersToTest:
        resultsMean, resultsVar = evaluateClassificationPerformance(classifierStr)
        plt.subplot(2, 2, jj)
        for ii in range(resultsMean.shape[0]):
            plt.plot(resultsMean[ii,:],label=miEstimators[ii])
        plt.legend()
        plt.title(classifierStr)

        jj = jj + 1

    plt.show()