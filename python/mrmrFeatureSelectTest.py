#!/usr/bin/env python

from sys import platform
import os
import pickle

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

miEstimators = ['ktau','knn_1','knn_6','knn_20','vme', 'ap']

NUM_CV = 10
SEED = 123
MAX_NUM_FEATURES = 50
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
        try:
            featureVec = sio.loadmat(os.path.join(folder,ds_lower,ds_lower+'_fs_'+miEstimator+'.mat'))
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None
        
    return (X_train,y_train,X_valid,y_valid,miFeatureSelections)    

def evaluateClassificationPerformance(classifierStr, dataset, enableCV=False):
    (X_train,y_train,X_valid,y_valid,miFeatureSelections) = readNips2003Data(dataset)
    
    y_train = np.squeeze(np.asarray(y_train))
    y_valid = np.squeeze(np.asarray(y_valid))

    resultsMean = np.empty((len(miEstimators),MAX_NUM_FEATURES))
    resultsVar = np.empty((len(miEstimators),MAX_NUM_FEATURES))
    resultsMean.fill(np.nan)
    resultsVar.fill(np.nan)

    eIdx = 0
    for estimator in miEstimators:
        featureVecAsList = miFeatureSelections[estimator]
        if(featureVecAsList is not None):
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
                    classifier = RandomForestClassifier(random_state=SEED,n_jobs=-2)
                elif(classifierStr=='KNN'):
                    classifier = KNeighborsClassifier()
                elif(classifierStr=='AdaBoost'):
                    classifier = AdaBoostClassifier(random_state=SEED)

                if(enableCV):
                    X = np.vstack((X_train,X_valid))
                    y = np.append(y_train,y_valid)
                    X_in = X[:,colSelect]
                    # normalize
                    X_in = preprocessing.scale(X_in)
                    scores = cross_val_score(classifier, X_in, y, cv=NUM_CV, n_jobs=-2)
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
                # print("numFeatures=%d Accuracy: %0.2f (+/- %0.2f)" % (ii, mu, sigma_sq))
                
                resultsMean[eIdx,ii-1] = mu
                resultsVar[eIdx,ii-1] = sigma_sq

        eIdx = eIdx + 1

    return (resultsMean,resultsVar)


if __name__=='__main__':
    datasetsToTest = ['Arcene','Dexter','Dorothea','Gisette','Madelon']
    classifiersToTest = ['SVC','RandomForest','KNN']
    enableCV = True

    if(enableCV):
        postPend = '_yesCV'
    else:
        postPend = '_noCV'

    resultsDir = os.path.join(folder, 'classification_results')

    for datasetIdx in range(len(datasetsToTest)):
        print('*'*10 + ' ' + datasetsToTest[datasetIdx] + ' ' + '*'*10)
        
        fig = plt.figure()
        datasetToTest = datasetsToTest[datasetIdx]
        jj = 1
        for classifierStr in classifiersToTest:
            fname = os.path.join(resultsDir,datasetToTest+'_'+classifierStr+postPend+'.pkl')
            if os.path.exists(fname):
                with open(fname,'rb') as f:
                    dataDict = pickle.load(f)
                resultsMean = dataDict['resultsMean']
                resultsVar  = dataDict['resultsVar']
            else:
                # only run if the results don't already exist
                resultsMean, resultsVar = evaluateClassificationPerformance(classifierStr,datasetToTest,enableCV)
                
            # store these results
            dataDict = {}
            dataDict['resultsMean'] = resultsMean
            dataDict['resultsVar']  = resultsVar
            with open(fname,'wb') as f:
                pickle.dump(dataDict,f)

            ax = fig.add_subplot(1, 3, jj)
            for ii in range(resultsMean.shape[0]):
                ax.plot(range(1,len(resultsMean[ii,:])+1),resultsMean[ii,:],label=miEstimators[ii])
                ax.set_xticks(np.arange(5,len(resultsMean[ii,:])+5,5))
            ax.legend()
            ax.set_title(classifierStr)

            jj = jj + 1

        fig.suptitle(datasetToTest)

    plt.show()