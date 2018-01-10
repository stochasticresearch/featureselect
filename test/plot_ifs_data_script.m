% plot the IFS data to understand the dependencies

clear;
clc;


% miEstimators = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap'};
% miEstimators = {'taukl','knn_1','knn_6','knn_20','vme','ap'};
% miEstimators = {'taukl','tau'};
depEstimators = {'taukl','tau','cim'};
miEstimators = {'knn_1','knn_6','knn_20','vme','ap'};
% miEstimators = {'ap'};

datasets = {'dexter','gisette','arcene','madelon'};

plot_ifs_data(depEstimators,datasets);
plot_ifs_data(miEstimators,datasets);