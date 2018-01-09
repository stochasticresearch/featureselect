% plot the IFS data to understand the dependencies

clear;
clc;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

% fNames = {'taukl','tau','knn_1','knn_6','knn_20','vme','ap'};
% fNames = {'taukl','knn_1','knn_6','knn_20','vme','ap'};
fNames = {'taukl','tau'};
datasets = {'dexter','gisette','arcene','madelon'};
% datasets = {'dexter','arcene','madelon'};

width = 30; height = width/5;
figure('paperpositionmode', 'auto', 'units', 'centimeters', 'position', [0 0 width height])

for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    subplot(1,4,dIdx);
    legendCell = cell(1,length(fNames));
    for fIdx=1:length(fNames)    
        fname = fNames{fIdx};

        dataFname = strcat(dataset,'_ifs_',fname,'.mat');
        fPath = fullfile(folder,dataset,dataFname);
        try
            load(fPath);

    %         [f,xi] = ksdensity(zscore(t));
            [f,xi] = ksdensity(t);
            plot(xi,f);
            hold on;

    %         if(fname=='taukl')
    %             fname='tau_{KL}';
    %         elseif(fname=='knn_20')
    %             fname='knn_{20}';
    %         end
            legendCell{fIdx} = fname;
        catch
        end
    end
    grid on;
    if(dIdx==4)
        legend(legendCell,'location','northeast');
    end
    title(dataset);
end
tightfig;