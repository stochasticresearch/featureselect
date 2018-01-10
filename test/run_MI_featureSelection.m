%% Generate matlab versions of the data so we can load fast
clear;
clc;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge\\arrhythmia';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge/arrhythmia';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge/arrhythmia';
end
inputFname = 'arrhythmia.data';
outputFname = 'arrhythmia.mat';
X = import_arrhythmia(fullfile(folder,inputFname));
save(fullfile(folder,outputFname),'X');

%% load the Arrhythmia data
clear;
clc;
dbstop if error;

% user configuration for which dataset we want to process
dataset = 'arrhythmia';

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

load(fullfile(folder,dataset,strcat(dataset,'.mat')));

% drop NA columns
y = X(:,end); X = X(:,1:end-1); 
[row, col] = find(isnan(X));
nancols = unique(col);
X(:,nancols) = [];

% save this matrix, which will actually be used with feature masks to test
% classification performance
if(~exist(fullfile(folder,dataset,'X.mat'),'file'))
    save(fullfile(folder,dataset,'X.mat'),'X','y');
end


%% load the rna-seq data
clear;
clc;
dbstop if error;

dataset = 'rnaseq';
if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

load(fullfile(folder,dataset,'data.mat'));
load(fullfile(folder,dataset,'labels.mat'));
y = double(labels');

[row, col] = find(isnan(X));
nancols = unique(col);
X(:,nancols) = [];

% save this matrix, which will actually be used with feature masks to test
% classification performance
if(~exist(fullfile(folder,dataset,'X.mat'),'file'))
    save(fullfile(folder,dataset,'X.mat'),'X','y');
end

%% Setup the NIPS Datasets data using routines provided

clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end
datasets = {'dexter','dorothea','arcene','gisette','madelon'};

for dataset=datasets
    [y_train, y_valid, y_test, X_train, X_valid, X_test] = ...
        read_nips2003_data(fullfile('/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge',dataset,dataset));
    X_train = full(X_train);
    X_valid = full(X_valid);
    save(fullfile(folder,dataset,'data.mat'), 'X_train', 'X_valid', 'X_test', 'y_train', 'y_valid', 'y_test');
end

%% Test the mRMR algorithm on various estimators of MI for different datasets
clear;
clc;
dbstop if error;

% setup the estimators of MI
knn_1 = 1;
knn_6 = 6;
knn_20 = 20;

functionHandlesCell = {@taukl_cc_mi_mex_interface;
                       @tau_mi_interface;
                       @cim_v2_hybrid;
                       {};
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @vmeMI_interface;
                       @apMI_interface;};

functionArgsCell    = {{0,1,0};
                       {};
                       {};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};};
fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap'};

datasets = {'dexter','dorothea','gisette','arcene','madelon'};

dispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');

for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    if(ispc)
        folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
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
    for ii=1:length(fNames)
        dispstat(sprintf('\t> Processing %s',fNames{ii}),'keepthis', 'timestamp');

        fs_outputFname = strcat(dataset,'_fs_',fNames{ii},'.mat');
        fOut = fullfile(folder,dataset,fs_outputFname);
        % if file exists, don't re-do it!
        if(~exist(fOut,'file'))
            tic;
            featureVec = mrmr_mid(X, y, numFeaturesToSelect, functionHandlesCell{ii}, functionArgsCell{ii});
            elapsedTime = toc;
            save(fOut,'featureVec','elapsedTime');
        end
    end
end

%% Get the mRMR algorithm's initial feature rankings for analysis
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

% setup the estimators of MI
knn_1 = 1;
knn_6 = 6;
knn_20 = 20;

functionHandlesCell = {@taukl_cc_mex_interface;
                       @corr;
                       @cim_v2_hybrid;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @vmeMI_interface;
                       @apMI_interface;};

functionArgsCell    = {{0,1,0};
                       {'type','kendall'};
                       {};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};};

fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap'};
datasets = {'dexter','dorothea','gisette','arcene','madelon'};

dispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');

for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    
    dispstat(sprintf('Processing %s',dataset),'keepthis', 'timestamp');

    load(fullfile(folder,dataset,'data.mat'));
    X = double(X_train);
    y = double(y_train);

    for ii=1:length(fNames)
        fs_outputFname = strcat(dataset,'_ifs_',fNames{ii},'.mat');
        fOut = fullfile(folder,dataset,fs_outputFname);
        dispstat(sprintf('\t> Processing %s',fNames{ii}),'keepthis', 'timestamp');
        % if file exists, don't re-do it!
        if(~exist(fOut,'file'))
            tic;
            t = mrmr_init_feature_ranking(X, y, functionHandlesCell{ii}, functionArgsCell{ii});
            elapsedTime = toc;
            save(fOut,'t','elapsedTime');
        end
    end
end


%% some tests to make sure that the serial and parallel versions of the

% % algorithm produce the same results
% testIdx = 1;
% K = 30;
% tic;
% 
% fea1 = mrmr_mid_serial(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
% fea2 = mrmr_mid(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
% % fea3 = mrmr_mid_parallel_old2(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
% % fprintf('1==2?=%d 1==3?=%d\n',isequal(fea1,fea2), isequal(fea1,fea3));
% fprintf('1==2?=%d\n',isequal(fea1,fea2));
% 
% % fea1 = mrmr_mid_debug(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
