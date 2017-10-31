#!/usr/bin/env python

from sys import platform
import os

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score

miEstimators = ['ktau','knn_1','knn_6','knn_20','vme','ap','cim']

numCV = 10
SEED = 12345
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

def evaluateSVC():
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
            clf = svm.SVC(random_state=SEED,max_iter=MAX_ITER)
            scores = cross_val_score(clf, X_in, y, cv=numCV)

            mu = scores.mean()
            sigma_sq = scores.std()*2
            print("numFeatures=%d Accuracy: %0.2f (+/- %0.2f)" % (ii, mu, sigma_sq))
            
            resultsMean[eIdx,ii-1] = mu
            resultsVar[eIdx,ii-1] = sigma_sq

        eIdx = eIdx + 1

    return (resultsMean,resultsVar)



if __name__=='__main__':
    resultsMean, resultsVar = evaluateSVC()
    for ii in range(resultsMean.shape[0]):
        plt.plot(resultsMean[ii,:],label=miEstimators[ii])
    plt.legend()
    plt.show()