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

miEstimators = ['cim','knn_1','knn_6','knn_20','vme', 'ap']

classifiersToTest = ['SVC','RandomForest','KNN']
datasetsToTest = ['mushrooms','phishing','crx','drivface']
    
NUM_CV = 10
SEED = 123
MAX_NUM_FEATURES = 50
MAX_ITER = 1000

folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','libsvm_datasets')

def readDataset(dataset):
    ds_lower = dataset.lower()
    z = sio.loadmat(os.path.join(folder,ds_lower+'_data.mat'))
    
    X = z['X']
    y = z['y']

    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            featureVec = sio.loadmat(os.path.join(folder,ds_lower+'_fs_'+miEstimator+'.mat'))
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None
        
    return (X,y,miFeatureSelections)    

def evaluateClassificationPerformance(classifierStr, dataset):
    (X,y,miFeatureSelections) = readDataset(dataset)
    
    y = np.squeeze(np.asarray(y))
    
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

                X_in = X[:,colSelect]
                # normalize
                X_in = preprocessing.scale(X_in)
                scores = cross_val_score(classifier, X_in, y, cv=NUM_CV, n_jobs=-2)
                
                mu = scores.mean()
                sigma_sq = scores.std()*2
                # print("numFeatures=%d Accuracy: %0.2f (+/- %0.2f)" % (ii, mu, sigma_sq))
                
                resultsMean[eIdx,ii-1] = mu
                resultsVar[eIdx,ii-1] = sigma_sq

        eIdx = eIdx + 1

    return (resultsMean,resultsVar)


if __name__=='__main__':
    
    resultsDir = os.path.join(folder, 'classification_results')
    try:
        os.makedirs(resultsDir)
    except:
        pass

    # run the ML
    for datasetIdx in range(len(datasetsToTest)):
        print('*'*10 + ' ' + datasetsToTest[datasetIdx] + ' ' + '*'*10)        
        datasetToTest = datasetsToTest[datasetIdx]
        for classifierStr in classifiersToTest:
            fname = os.path.join(resultsDir,datasetToTest+'_'+classifierStr+'.pkl')
            if os.path.exists(fname):
                with open(fname,'rb') as f:
                    dataDict = pickle.load(f)
                resultsMean = dataDict['resultsMean']
                resultsVar  = dataDict['resultsVar']
            else:
                # only run if the results don't already exist
                resultsMean, resultsVar = evaluateClassificationPerformance(classifierStr,datasetToTest)
                
            # store these results
            dataDict = {}
            dataDict['resultsMean'] = resultsMean
            dataDict['resultsVar']  = resultsVar
            with open(fname,'wb') as f:
                pickle.dump(dataDict,f)

    # plot the stuff & store
    estimatorsLegend = map(lambda x:x.upper(),miEstimators)
    estimatorsLegend[estimatorsLegend.index('TAUKL')]  = r'$\tau_{KL}$'
    if('tau' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('TAU')]  = r'$\tau$'
    
    for dataset in datasetsToTest:
        try:
            os.makedirs(os.path.join(resultsDir,'figures'))
        except:
            pass
        outputFname = os.path.join(resultsDir,'figures',dataset+'.png')
        
        fig,ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(9,3))

        yMinVal = 1.0
        yMaxVal = 0.0
        for cIdx in range(len(classifiersToTest)):
            classifier = classifiersToTest[cIdx]
            f = os.path.join(resultsDir,dataset+'_'+classifier+'.pkl')
            with open(f,'rb') as f:
                z = pickle.load(f)

            lineHandlesVec = []
            for estimatorIdx in range(z['resultsMean'].shape[0]):
                resultsMean = z['resultsMean'][estimatorIdx,:]
                results2Var = z['resultsVar'][estimatorIdx,:]
                resultsStd = np.sqrt(results2Var/2.)
                xx = range(1,len(resultsMean)+1)

                y = resultsMean
                h = ax[cIdx].plot(xx, y)
                
                yLo = resultsMean-results2Var/2.
                yHi = resultsMean+results2Var/2.
                ax[cIdx].fill_between(xx, yLo, yHi, alpha=0.2)
                ax[cIdx].grid(True)
                ax[cIdx].set_xticks([10,30,50])
                lineHandlesVec.append(h[0])
                
                if(min(yLo)<yMinVal):
                    yMinVal = min(yLo)
                if(max(yHi)>yMaxVal):
                    yMaxVal = max(yHi)
            ax[cIdx].set_title(classifier)
            if(cIdx==0):
                ax[cIdx].set_ylabel(dataset.upper()+'\nClassification Accuracy')
            if(cIdx==1):
                ax[cIdx].set_xlabel('# Features')
            
        plt.figlegend( lineHandlesVec, estimatorsLegend, loc = 'center right' )
        plt.savefig(outputFname, bbox_inches='tight')
