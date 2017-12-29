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
#miEstimators = ['ktau','knn_1','knn_6','knn_20','ap','cim']
#miEstimators = ['ktau','knn_1','knn_6','knn_20','vme', 'ap']
#miEstimators = ['ktau','knn_6','knn_20','vme', 'ap']
#miEstimators = ['ktau','knn_6','knn_20','ap']
miEstimators = ['ktau','ap']

numCV = 10
SEED = 123
MAX_NUM_FEATURES = 20
MAX_ITER = 1000

folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','feature_select_challenge')

def readNips2003Data(dataset):
    ds_lower = dataset.lower()
    z = sio.loadmat(os.path.join(folder,ds_lower,'data.mat'))
    
    X_train = z['X_train']
    y_train = z['y_train']
    X_valid = z['X_valid']
    y_valid = z['y_valid']

    miFeatureSelections = {}
    for miEstimator in miEstimators:
        featureVec = sio.loadmat(os.path.join(folder,ds_lower,ds_lower+'_fs_'+miEstimator+'.mat'))
        miFeatureSelections[miEstimator] = featureVec['featureVec']
    
    return (X_train,y_train,X_valid,y_valid,miFeatureSelections)    

def evaluateClassificationPerformance(classifierStr, dataset):
    (X_train,y_train,X_valid,y_valid,miFeatureSelections) = readNips2003Data(dataset)
    
    y_train = np.squeeze(np.asarray(y_train))
    y_valid = np.squeeze(np.asarray(y_valid))
    numCV_val = 1  # we don't do cross-validation here b/c the data is split between
                   # train/test/validation already, so we just test on the validation
                   # set directly w/out CV

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

            if(numCV_val>1):
                X_in = X[:,colSelect]
                # normalize
                X_in = preprocessing.scale(X_in)
                scores = cross_val_score(classifier, X_in, y, cv=numCV, n_jobs=-2)
            else:
                X_train_in = X_train[:,colSelect]
                X_valid_in = X_valid[:,colSelect]
                # normalize
                X_train_in = preprocessing.scale(X_train_in)
                X_valid_in = preprocessing.scale(X_valid_in)

                clf = classifier.fit(X_train_in,y_train)
                scores = np.asarray([clf.score(X_valid_in,y_valid)])

            mu = scores.mean()
            sigma_sq = scores.std()*2
            print("numFeatures=%d Accuracy: %0.2f (+/- %0.2f)" % (ii, mu, sigma_sq))
            
            resultsMean[eIdx,ii-1] = mu
            resultsVar[eIdx,ii-1] = sigma_sq

        eIdx = eIdx + 1

    return (resultsMean,resultsVar)


if __name__=='__main__':
    datasetsToTest = ['Arcene','Dexter','Dorothea','Gisette','Madelon']
    classifiersToTest = ['SVC','RandomForest','KNN']

    for datasetIdx in range(len(datasetsToTest)):
        fig = plt.figure()

        datasetToTest = datasetsToTest[datasetIdx]
        jj = 1
        for classifierStr in classifiersToTest:
            resultsMean, resultsVar = evaluateClassificationPerformance(classifierStr,datasetToTest)
            ax = fig.add_subplot(1, 3, jj)
            for ii in range(resultsMean.shape[0]):
                ax.plot(range(1,len(resultsMean[ii,:])+1),resultsMean[ii,:],label=miEstimators[ii])
                ax.set_xticks(np.arange(2,len(resultsMean[ii,:])+1,4))
            ax.legend()
            ax.set_title(classifierStr)

            jj = jj + 1

        fig.suptitle(datasetToTest)

    plt.show()