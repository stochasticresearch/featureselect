%% Read in other dataset and get an initial assessment of how Tau values are distributed
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\Desktop\\new_datasets';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/libsvm_datasets';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/libsvm_datasets';
end

functionHandlesCell = {@taukl_cc_mex_interface;
                       @corr;};
functionArgsCell    = {{0,1,0};
                       {'type','kendall'}};
fNames = {'taukl','tau'};

datasets = {'mushrooms','phishing'};
numFeaturesToSelect = 50;

for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    [y,X] = libsvmread(fullfile(folder,dataset));
    X = full(X);

    for ii=1:length(fNames)
        fs_outputFname = strcat(dataset,'_ifs_',fNames{ii},'.mat');
        fOut = fullfile(folder,fs_outputFname);
        dispstat(sprintf('\t> [IFS] Processing %s',fNames{ii}),'keepthis', 'timestamp');
        % if file exists, don't re-do it!
        if(~exist(fOut,'file'))
            tic;
            t = mrmr_init_feature_ranking(X, y, functionHandlesCell{ii}, functionArgsCell{ii});
            elapsedTime = toc;
            save(fOut,'t','elapsedTime');
        end
        
        dispstat(sprintf('\t> [FS] Processing %s',fNames{ii}),'keepthis', 'timestamp');
        fs_outputFname = strcat(dataset,'_fs_',fNames{ii},'.mat');
        fOut = fullfile(folder,fs_outputFname);
        if(~exist(fOut,'file'))
            tic;
            featureVec = mrmr_mid(X, y, numFeaturesToSelect, functionHandlesCell{ii}, functionArgsCell{ii});
            elapsedTime = toc;
            save(fOut,'featureVec','elapsedTime');
        end
    end
end

%% Explore the IFS files for Mushroom
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\Desktop\\new_datasets';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/libsvm_datasets';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/libsvm_datasets';
end

dataset = 'mushrooms';

miEstimator = 'taukl';
dataFname = strcat(dataset,'_ifs_',miEstimator,'.mat');
fPath = fullfile(folder,dataFname);
load(fPath);
tTauKL = t;

miEstimator = 'tau';
dataFname = strcat(dataset,'_ifs_',miEstimator,'.mat');
fPath = fullfile(folder,dataFname);
load(fPath);
tTau = t;

h1 = histogram(tTauKL);
hold on;
h2 = histogram(tTau);

h1.Normalization = 'probability';
h1.BinWidth = 0.1;
h2.Normalization = 'probability';
h2.BinWidth = 0.1;
title('Mushrooms');

%% Explore the IFS files for Phishing
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\Desktop\\new_datasets';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/libsvm_datasets';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/libsvm_datasets';
end

dataset = 'phishing';

miEstimator = 'taukl';
dataFname = strcat(dataset,'_ifs_',miEstimator,'.mat');
fPath = fullfile(folder,dataFname);
load(fPath);
tTauKL = t;

miEstimator = 'tau';
dataFname = strcat(dataset,'_ifs_',miEstimator,'.mat');
fPath = fullfile(folder,dataFname);
load(fPath);
tTau = t;

h1 = histogram(tTauKL);
hold on;
h2 = histogram(tTau);

h1.Normalization = 'probability';
h1.BinWidth = 0.1;
h2.Normalization = 'probability';
h2.BinWidth = 0.1;

title('Phishing');