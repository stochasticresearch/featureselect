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

import pandas as pd

miEstimators = ['cim','knn_1','knn_6','knn_20','vme', 'ap']

classifiersToTest = ['SVC','RandomForest','KNN']
datasetsToTest = ['Arcene','Dexter','Dorothea','Madelon','drivface','rf_fingerprinting']
#datasetsToTest = ['Arcene','Dexter','Dorothea','Madelon','drivface','mushrooms','phishing']
    
NUM_CV = 10
SEED = 123
MAX_NUM_FEATURES = 50
MAX_ITER = 1000

figures_folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','feature_selection_results_figures')

def getDataFolder(dataset):
    dsl = dataset.lower()
    if(dsl=='arcene' or 
       dsl=='dexter' or 
       dsl=='dorothea' or 
       dsl=='madelon'):
        folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','feature_select_challenge')
    elif(dsl=='drivface' or
         dsl=='mushrooms' or
         dsl=='phishing'):
        folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results','drivface')
    elif(dsl=='rf_fingerprinting'):
        folder = os.path.join(os.environ['HOME'],'ownCloud','PhD','sim_results',
            'rf_fingerprinting','data','fs_results')
    return folder

def readDataset(dataset):
    dsl = dataset.lower()
    if(dsl=='arcene' or 
       dsl=='dexter' or 
       dsl=='dorothea' or 
       dsl=='madelon'):
        return _readNips2003Data(dataset)
    elif(dsl=='drivface' or
         dsl=='mushrooms' or
         dsl=='phishing'):
        return _readLibsvmDatasets(dataset)
    elif(dsl=='rf_fingerprinting'):
        return _readRfFingerprintingDatasets()

def _readRfFingerprintingDatasets():
    # static configuration ... do we need to parametrize?
    numQ = 3
    numSamplesForQ = 200

    # read the data
    fName = 'data_numQ_%d_numSampsForQ_%d.csv' % (numQ,numSamplesForQ)
    df = pd.read_csv(os.path.join(getDataFolder('rf_fingerprinting'),'..',fName))

    feature_cols = df.columns.tolist()
    feature_cols.remove('transmitter_id')
    X = df[feature_cols].values
    y = df['transmitter_id'].values

    # read  feature selection vectors
    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            matFile = 'data_numQ_%d_numSampsForQ_%d.csv_fs_%s.mat' % (numQ,numSamplesForQ,miEstimator)
            matFileAndPath = os.path.join(getDataFolder('rf_fingerprinting'),matFile)
            featureVec = sio.loadmat(matFileAndPath)
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None

    return (X,y,miFeatureSelections)

def _readLibsvmDatasets(dataset):
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

def _readNips2003Data(dataset):
    ds_lower = dataset.lower()
    z = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,'data.mat'))
    
    X_train = z['X_train']
    y_train = z['y_train']
    X_valid = z['X_valid']
    y_valid = z['y_valid']

    y_train = np.squeeze(np.asarray(y_train))
    y_valid = np.squeeze(np.asarray(y_valid))
    X = np.vstack((X_train,X_valid))
    y = np.append(y_train,y_valid)
    
    miFeatureSelections = {}
    for miEstimator in miEstimators:
        try:
            featureVec = sio.loadmat(os.path.join(getDataFolder(dataset),ds_lower,ds_lower+'_fs_'+miEstimator+'.mat'))
            miFeatureSelections[miEstimator] = featureVec['featureVec']
        except:
            miFeatureSelections[miEstimator] = None
    
    return (X,y,miFeatureSelections)

def evaluateClassificationPerformance(classifierStr, dataset):
    (X,y,miFeatureSelections) = readDataset(dataset)

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
                
                resultsMean[eIdx,ii-1] = mu
                resultsVar[eIdx,ii-1] = sigma_sq

        eIdx = eIdx + 1

    return (resultsMean,resultsVar)


if __name__=='__main__':
    # run the ML
    for datasetIdx in range(len(datasetsToTest)):
        print('*'*10 + ' ' + datasetsToTest[datasetIdx] + ' ' + '*'*10)        
        datasetToTest = datasetsToTest[datasetIdx]
        resultsDir = os.path.join(getDataFolder(datasetToTest), 'classification_results')
        try:
            os.makedirs(resultsDir)
        except:
            pass
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
    try:
        estimatorsLegend[estimatorsLegend.index('TAUKL')]  = r'$\tau_{KL}$'
    except:
        pass
    if('tau' in miEstimators):
        estimatorsLegend[estimatorsLegend.index('TAU')]  = r'$\tau$'
    estimatorsLegend[estimatorsLegend.index('VME')]    = 'vME'
    estimatorsLegend[estimatorsLegend.index('KNN_1')]  = r'$KNN_1$'
    estimatorsLegend[estimatorsLegend.index('KNN_6')]  = r'$KNN_6$'
    estimatorsLegend[estimatorsLegend.index('KNN_20')] = r'$KNN_{20}$'

    for dataset in datasetsToTest:
        resultsDir = os.path.join(getDataFolder(dataset),'classification_results')
        outputFname = os.path.join(figures_folder,dataset+'.png')
        
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
        # because of a wide variance for KTAU w/ the first feature for Dorothea, 
        # we have to manually set yMin and yMax, otherwise plot is uninformative
        if(dataset=='Dorothea'):
            ax[cIdx].set_ylim(0.8,1.0)
            
        plt.figlegend( lineHandlesVec, estimatorsLegend, loc = 'center right' )
        plt.savefig(outputFname, bbox_inches='tight')
