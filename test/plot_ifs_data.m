function [] = plot_ifs_data(miEstimators,datasets)

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

% width = 15; height = 15;
% figure('paperpositionmode', 'auto', 'units', 'centimeters', 'position', [0 0 width height])
figure;

ksdensity_buf = 0.05;
ksdensity_bw = 0.5;
for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    subplot(2,2,dIdx);
    legendCell = cell(1,length(miEstimators));
    for fIdx=1:length(miEstimators)    
        fname = miEstimators{fIdx};

        dataFname = strcat(dataset,'_ifs_',fname,'.mat');
        fPath = fullfile(folder,dataset,dataFname);
        try
            load(fPath);
            
            if(strcmpi(fname,'taukl') || strcmpi(fname,'tau') || strcmpi(fname,'cim'))
%                 [f,xi] = ksdensity(t);
                tt = dep2mi(t);
                [f,xi] = ksdensity(tt,'Support',[-ksdensity_buf max(tt)+ksdensity_buf],'Bandwidth',ksdensity_bw);
            else
                [f,xi] = ksdensity(t,'Support',[-ksdensity_buf max(t)+ksdensity_buf],'Bandwidth',ksdensity_bw);
            end
            plot(xi,f);
            hold on;

            legendCell{fIdx} = fname;
        catch ME
            fname
            dataset
            ME
        end
    end
    grid on;
    if(dIdx==4)
        legend(legendCell,'location','northeast');
    end
    title(dataset);
end
% tightfig;

end