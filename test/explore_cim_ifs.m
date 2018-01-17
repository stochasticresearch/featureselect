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

dataset = 'dexter';

% load the data
load(fullfile(folder,dataset,'data.mat'));
X = double(X_train);
y = double(y_train);

% load IFS for TAU
miEstimator = 'taukl';
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

subplot(2,3,1); plot(tTau); title('\tau_{KL}'); 
subplot(2,3,2); plot(tCim); title('CIM');
subplot(2,3,3); plot(abs(tTau)-abs(tCim)); title('|\tau_{KL}|-|CIM|');
subplot(2,3,4); plot(dep2mi(tTau)); title('MI(\tau_{KL})');
subplot(2,3,5); plot(dep2mi(tCim)); title('MI(cim)');
subplot(2,3,6); plot(dep2mi(tTau)-dep2mi(tCim)); title('MI(\tau_{KL})-MI(CIM)');

%% explore why CIM values for several interesting indices in the arcene dataset!

clear;
clc;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

dataset = 'dexter';

% load the data
load(fullfile(folder,dataset,'data.mat'));
X = double(X_train);
y = double(y_train);

idxOfInterest = 626;

xx = X(:,idxOfInterest);
yy = y;

[u,v] = pobs_sorted_cc_mex(xx,yy); 

abs(taukl_cc(u,v,0,1,0))
abs(taukl_cc(v,u,0,1,0))
cim_v2_hybrid_cc(u,v,0.015625,0.2)