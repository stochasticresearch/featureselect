% explore why we are getting NaN's for tau

clear;
clc;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

fHandle = @corr;
fArgs = {'type','kendall'};

dataset = 'dexter';

load(fullfile(folder,dataset,'data.mat'));
X = double(X_train);
y = double(y_train);

t = mrmr_init_feature_ranking(X, y, fHandle, fArgs);