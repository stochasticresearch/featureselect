%% run MI feature selection for RNASEQ data

clear;
clc;
dbstop if error;

% load X & y from dataset
folder='/Users/karrak1/Documents/erc_paper/old_simulations/feature_select_challenge/not_nips/rnaseq';
load(fullfile(folder, 'X.mat'));  % loads variables X, y

% split into train/test (80/20)
cv = cvpartition(size(X,1),'HoldOut',0.2);
idx = cv.test;
X_train = X(~idx,:);
y_train = y(~idx,:);
X_valid = X(idx,:);
y_valid = y(idx,:);
% save this as a X_all.mat
save(fullfile(folder, 'X_all.mat'), 'X_train', 'y_train', 'X_valid', 'y_valid');

dataset = 'rnaseq';
fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap','h_mi',...
    'dCor','MIC','corr','RDC'};
algosToRun = {'cim','knn_1','knn_6','knn_20','vme','ap','h_mi'};

% setup the estimators of MI
knn_1 = 1; knn_6 = 6; knn_20 = 20;
msi = 0.015625; alpha = 0.2; 
mine_c = 15; mine_alpha = 0.6;
rdc_k = 20; rdc_s = 1/6;

functionHandlesCell = {@taukl_cc_mi_mex_interface;
                       @tau_mi_interface;
                       @cim;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @vmeMI_interface;
                       @apMI_interface;
                       @h_mi_interface;
                       @dcor;
                       @mine_interface_mic;
                       @corr;
                       @rdc;};
autoDetectHybrid = 0; isHybrid = 1; continuousRvIndicator = 0;
functionArgsWithOutputCell    = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {1};
                       {};
                       {mine_alpha,mine_c,'mic_e'};
                       {};
                       {rdc_k, rdc_s};};
autoDetectHybrid = 0; isHybrid = 0; continuousRvIndicator = 0;
functionArgsInterDepCell    = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {1};
                       {};
                       {mine_alpha,mine_c,'mic_e'};
                       {};
                       {rdc_k, rdc_s};};

% we don't do splitting between 50 & 75% here, b/c this is multi-class
dispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');
load(fullfile(folder, 'X_all.mat'));

numFeaturesToSelect = 50;
for ii=1:length(algosToRun)
    accessIdx = find(contains(algosToRun{ii},fNames));
    dispstat(sprintf('\t> Processing %s',fNames{accessIdx}),'keepthis', 'timestamp');
    fs_outputFname = strcat(dataset,'_fs_',fNames{accessIdx},'.mat');

    fOut = fullfile(folder, dataset, 'fs_results', fs_outputFname);
    % if file exists, don't re-do it!
    if(~exist(fOut,'file'))
        tic;
        featureVec = mrmr_mid(X_train, y_train, numFeaturesToSelect, ...
            functionHandlesCell{accessIdx}, ...
            functionArgsWithOutputCell{accessIdx}, ...
            functionArgsInterDepCell{accessIdx});
        elapsedTime = toc;
        save(fOut,'featureVec','elapsedTime');
    end
end
