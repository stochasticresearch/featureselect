%% Test the mRMR algorithm on various estimators of MI for different datasets
clear;
clc;
dbstop if error;

% setup the estimators of MI
knn_1 = 1;
knn_6 = 6;
knn_20 = 20;
msi = 0.015625; alpha = 0.2; 
autoDetectHybrid = 0; isHybrid = 1; continuousRvIndicator = 0;

functionHandlesCell = {@taukl_cc_mi_mex_interface;
                       @tau_mi_interface;
                       @cim;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @vmeMI_interface;
                       @apMI_interface;};

functionArgsCell    = {{0,1,0};
                       {};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};};
fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap'};

dataset = 'arcene';
if(ispc)
False    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end
dispstat(sprintf('Processing %s',dataset),'keepthis', 'timestamp');

load(fullfile(folder,dataset,'data.mat'));
X = double(X_train);
y = double(y_train);

numFeaturesToSelect = 50;
K_MAX = 1000;
featureVec = mrmr_mid_testing(X, y, numFeaturesToSelect, functionHandlesCell{2}, functionArgsCell{2},K_MAX);
    