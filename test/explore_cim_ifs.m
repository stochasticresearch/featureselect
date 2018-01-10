%% explore the difference in the initial feature ranking (IFS) for CIM vs. TAU

clear;
clc;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

dataset = 'arcene';

% load the data
load(fullfile(folder,dataset,'data.mat'));
X = double(X_train);
y = double(y_train);

% load IFS for TAU
miEstimator = 'tau';
dataFname = strcat(dataset,'_ifs_',miEstimator,'.mat');
fPath = fullfile(folder,dataset,dataFname);
load(fPath);
tTau = t;

% load IFS for CIM
miEstimator = 'cim';
dataFname = strcat(dataset,'_ifs_',miEstimator,'.mat');
fPath = fullfile(folder,dataset,dataFname);
load(fPath);
tCim = t;

subplot(2,3,1); plot(tTau); title('\tau'); 
subplot(2,3,2); plot(tCim); title('CIM');
subplot(2,3,3); plot(abs(tTau)-abs(tCim)); title('|\tau|-|CIM|');
subplot(2,3,4); plot(dep2mi(tTau)); title('MI(\tau)');
subplot(2,3,5); plot(dep2mi(tCim)); title('MI(cim)');
subplot(2,3,6); plot(dep2mi(tTau)-dep2mi(tCim)); title('MI(\tau)-MI(CIM)');

%% explore why CIM is producing 0.5 consantly for several indices in the arcene dataset!

clear;
clc;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

dataset = 'arcene';

% load the data
load(fullfile(folder,dataset,'data.mat'));
X = double(X_train);
y = double(y_train);

% load IFS for CIM
miEstimator = 'cim';
dataFname = strcat(dataset,'_ifs_',miEstimator,'.mat');
fPath = fullfile(folder,dataset,dataFname);
load(fPath);
tCim = t;

xIdxOfInterest = find(tCim==0.5);