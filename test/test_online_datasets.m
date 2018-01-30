%% Test the online learning datasets
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\Desktop\\online_learning_datasets';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/online_learning_datasets';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/online_learning_datasets';
end

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

X = load_from_infra(fullfile(folder,'crx.db'));
y = load(fullfile(folder,'crx.ascii'),'-ascii');

numFeaturesToSelect = 50;

for ii=1:length(fNames)
    dispstat(sprintf('\t> [FS] Processing %s',fNames{ii}),'keepthis', 'timestamp');
    fs_outputFname = strcat('crx_fs_',fNames{ii},'.mat');
    fOut = fullfile(folder,fs_outputFname);
    if(~exist(fOut,'file'))
        tic;
        featureVec = mrmr_mid(X, y, numFeaturesToSelect, functionHandlesCell{ii}, functionArgsCell{ii});
        elapsedTime = toc;
        save(fOut,'featureVec','elapsedTime');
    end
end