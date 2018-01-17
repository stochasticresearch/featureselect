%% Test the DrivFace dataset
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\Desktop\\drivface';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/drivface';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/drivface';
end

functionHandlesCell = {@taukl_cc_mi_mex_interface;
                       @tau_mi_interface;
                       @cim_v2_hybrid;
                       {};
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @vmeMI_interface;
                       @apMI_interface;
                       @cim_v2_hybrid_mi};

functionArgsCell    = {{0,1,0};
                       {};
                       {};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {};};
fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap','cim_mi'};

load(fullfile(folder,'DrivFace.mat'));

X = drivFaceD.data;
y = drivFaceD.nlab;

numFeaturesToSelect = 50;

for ii=1:length(fNames)
    dispstat(sprintf('\t> [FS] Processing %s',fNames{ii}),'keepthis', 'timestamp');
    fs_outputFname = strcat('drivface_fs_',fNames{ii},'.mat');
    fOut = fullfile(folder,fs_outputFname);
    if(~exist(fOut,'file'))
        tic;
        featureVec = mrmr_mid(X, y, numFeaturesToSelect, functionHandlesCell{ii}, functionArgsCell{ii});
        elapsedTime = toc;
        save(fOut,'featureVec','elapsedTime');
    end
end