%% Generate the scores for the selected features
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\synthetic_feature_select';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/synthetic_feature_select';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/synthetic_feature_select';
end

% the configuration we want to score

numIndependentFeatures = 20;
numRedundantFeatures = 0;
numUselessFeatures = 80;
skews = {'left_skew','no_skew','right_skew'};
dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
fNames = {'taukl','cim','knn_1','knn_6','knn_20','ap','h_mi'};
numSamps = 100;
numMCSims = 25;

% setup output filename
inputFname = sprintf('res_%d_%d_%d_%d_%d_timesOpOnly.mat',...
    numIndependentFeatures,numRedundantFeatures,numUselessFeatures,numSamps,numMCSims);
load(fullfile(folder,inputFname))

% bar plot configuration
numBars = length(fNames);
numGroups = length(dep_clusters);
width = 1;
groupnames = {'Low','Med','Hi','All'};
titles = {'Left-Skew','No-Skew','Right-Skew'};
bw_xlabel = [];
bw_ylabel = [];
bw_color_map = parula;
gridstatus = 'y';
bw_legend_val = {'\tau_{KL}','CIM','KNN-1','KNN-6','KNN-20','AP','H_{MI}'};
error_sides = 2;
legend_type = 'plot';
legendTextSize = 17;
labelTextSize = 20;
groupTextSize = 20;

for skIdx=1:length(skews)
    sk = skews{skIdx};
    barMatrix_val = zeros(numGroups,numBars);
    barMatrix_err = zeros(numGroups,numBars);
    bw_title = sk;
    for dcIdx=1:length(dep_clusters)
        dc = dep_clusters{dcIdx};
        fprintf('***** %s-%s *****\n',sk,dc);
        for fIdx=1:length(fNames)
            estimator = fNames{fIdx};
            % get the selected matrix
            X = selectedFeaturesResultsMap(sk,dc,estimator);
%             score_vec = score_synthetic_fs(X,numIndependentFeatures,numRedundantFeatures,numUselessFeatures);
            score_vec = score_synthetic_fs_v2(X,numIndependentFeatures,numRedundantFeatures);
            fprintf('\t %s-->[%0.02f,%0.02f]\n',estimator,mean(score_vec),std(score_vec));
            barMatrix_val(dcIdx,fIdx) = mean(score_vec);
            barMatrix_err(dcIdx,fIdx) = std(score_vec)/2;
        end
    end
    subplot(1,3,skIdx);
    if(skIdx==length(skews))
        bw_legend = bw_legend_val;
    else
        bw_legend = [];
    end
% 	figure;
%     bw_legend = bw_legend_val;
%     bw_legend = '';
    bwo = barweb(barMatrix_val,barMatrix_err,width,groupnames,bw_title,bw_xlabel,bw_ylabel,...
        bw_color_map,gridstatus,bw_legend,error_sides,legend_type,...
        legendTextSize, labelTextSize, groupTextSize);
%     set(gca, 'xticklabel', groupnames, ...
%         'box', 'off', ...
%         'ticklength', [0 0], ...
%         'fontsize', groupTextSize, ...
%         'xtick',1:length(groupnames), ...
%         'linewidth', 2, ...
%         'xaxisLocation','top', ...
%         'xgrid','on', ...
%         'ygrid','on');
%     '/', '\', '|', '-', '+', 'x', '.', 'c', 'w', 'k'
%     im_hatch = applyhatch_pluscolor(gcf,'/\|-+x',0);

      % save mat files to plot in Python
%     f_out = fullfile(folder, titles{skIdx});
%     save(f_out,'barMatrix_val','barMatrix_err','groupnames','bw_legend_val');

end


%% load and plot the results of the pairwise mapping

clear;
clc;
dbstop if error;
rng(123);

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\synthetic_feature_select';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/synthetic_feature_select';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/synthetic_feature_select';
end

numIndependentFeatures = 20;
numRedundantFeatures = 0;
numUselessFeatures = 80;
% skews = {'left_skew','no_skew','right_skew'};
% dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
% fNames = {'taukl','cim','knn_1','knn_6','knn_20','ap','h_mi'};
numSamps = 100;
numMCSims = 25;

ffname = 'plusOpOnly';
inputFname = sprintf('res_%d_%d_%d_%d_%d_%s.mat',...
    numIndependentFeatures,numRedundantFeatures,numUselessFeatures,numSamps,numMCSims,ffname);
load(fullfile(folder,inputFname))
loadMapName = depWithOutputResultsMap;

% create a matrix to compare each of the methods over all the MC
% simulations
cmap = helper_power_colormap();

fillVal = 0;
filler = fillVal*ones(1,200);
yLabelLegend = {'H_{MI}','CIM','KNN-1','KNN-6','KNN-20','AP'};

subplot(4,3,1);
nsk_all = [mean(loadMapName('no_skew','all_cluster','h_mi'),1); ...
           mean(abs(loadMapName('no_skew','all_cluster','cim')),1); ...
           mean(abs(loadMapName('no_skew','all_cluster','knn_1')),1); ...
           mean(abs(loadMapName('no_skew','all_cluster','knn_6')),1); ...
           mean(abs(loadMapName('no_skew','all_cluster','knn_20')),1); ...
           mean(abs(loadMapName('no_skew','all_cluster','ap')),1); ...
                   ];
plotdata = nsk_all./max(nsk_all,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
hold on;
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/All');

subplot(4,3,2);
imagesc(squeeze(mean(interDepResultsMap('no_skew','all_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,3);
imagesc(squeeze(mean(interDepResultsMap('no_skew','all_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,4);
nsk_lo = [mean(loadMapName('no_skew','lo_cluster','h_mi'),1); ...
          mean(abs(loadMapName('no_skew','lo_cluster','cim')),1); ...
          mean(abs(loadMapName('no_skew','lo_cluster','knn_1')),1); ...
          mean(abs(loadMapName('no_skew','lo_cluster','knn_6')),1); ...
          mean(abs(loadMapName('no_skew','lo_cluster','knn_20')),1); ...
          mean(abs(loadMapName('no_skew','lo_cluster','ap')),1); ...
                   ];
plotdata = nsk_lo./max(nsk_lo,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/Lo');

subplot(4,3,5);
imagesc(squeeze(mean(interDepResultsMap('no_skew','lo_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,6);
imagesc(squeeze(mean(interDepResultsMap('no_skew','lo_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,7);
nsk_med = [mean(loadMapName('no_skew','med_cluster','h_mi'),1); ...
           mean(abs(loadMapName('no_skew','med_cluster','cim')),1); ...
           mean(abs(loadMapName('no_skew','med_cluster','knn_1')),1); ...
           mean(abs(loadMapName('no_skew','med_cluster','knn_6')),1); ...
           mean(abs(loadMapName('no_skew','med_cluster','knn_20')),1); ...
           mean(abs(loadMapName('no_skew','med_cluster','ap')),1); ...
                   ];
plotdata = nsk_med./max(nsk_med,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/Med');

subplot(4,3,8);
imagesc(squeeze(mean(interDepResultsMap('no_skew','med_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,9);
imagesc(squeeze(mean(interDepResultsMap('no_skew','med_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,10);
nsk_hi = [mean(loadMapName('no_skew','hi_cluster','h_mi'),1); ...
          mean(abs(loadMapName('no_skew','hi_cluster','cim')),1); ...
          mean(abs(loadMapName('no_skew','hi_cluster','knn_1')),1); ...
          mean(abs(loadMapName('no_skew','hi_cluster','knn_6')),1); ...
          mean(abs(loadMapName('no_skew','hi_cluster','knn_20')),1); ...
          mean(abs(loadMapName('no_skew','hi_cluster','ap')),1); ...
                   ];
plotdata = nsk_hi./max(nsk_hi,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/Hi');

subplot(4,3,11);
imagesc(squeeze(mean(interDepResultsMap('no_skew','hi_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,12);
imagesc(squeeze(mean(interDepResultsMap('no_skew','hi_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

hh = get(subplot(4,3,12),'Position');
cc = colorbar('Position', [hh(1)+hh(3)+.03  hh(2)+.03  0.02  hh(2)+hh(3)*3]);
set(cc,'fontsize',10, 'ytick',[0, 0.25, .5, 0.75, 1], ...
    'yticklabel', {'0.0', '0.25', '0.50', '0.75', '1.0'}, 'linewidth', 0.5);

figure;
subplot(4,3,1);
lsk_all = [mean(loadMapName('left_skew','all_cluster','h_mi'),1); ...
           mean(abs(loadMapName('left_skew','all_cluster','cim')),1); ...
           mean(abs(loadMapName('left_skew','all_cluster','knn_1')),1); ...
           mean(abs(loadMapName('left_skew','all_cluster','knn_6')),1); ...
           mean(abs(loadMapName('left_skew','all_cluster','knn_20')),1); ...
           mean(abs(loadMapName('left_skew','all_cluster','ap')),1); ...
                   ];
plotdata = lsk_all./max(lsk_all,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/All');

subplot(4,3,2);
imagesc(squeeze(mean(interDepResultsMap('left_skew','all_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,3);
imagesc(squeeze(mean(interDepResultsMap('left_skew','all_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,4);
lsk_lo = [mean(loadMapName('left_skew','lo_cluster','h_mi'),1); ...
          mean(abs(loadMapName('left_skew','lo_cluster','cim')),1); ...
          mean(abs(loadMapName('left_skew','lo_cluster','knn_1')),1); ...
          mean(abs(loadMapName('left_skew','lo_cluster','knn_6')),1); ...
          mean(abs(loadMapName('left_skew','lo_cluster','knn_20')),1); ...
          mean(abs(loadMapName('left_skew','lo_cluster','ap')),1); ...
                   ];
plotdata = lsk_lo./max(lsk_lo,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/Lo');

subplot(4,3,5);
imagesc(squeeze(mean(interDepResultsMap('left_skew','lo_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,6);
imagesc(squeeze(mean(interDepResultsMap('left_skew','lo_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,7);
lsk_med = [mean(loadMapName('left_skew','med_cluster','h_mi'),1); ...
           mean(abs(loadMapName('left_skew','med_cluster','cim')),1); ...
           mean(abs(loadMapName('left_skew','med_cluster','knn_1')),1); ...
           mean(abs(loadMapName('left_skew','med_cluster','knn_6')),1); ...
           mean(abs(loadMapName('left_skew','med_cluster','knn_20')),1); ...
           mean(abs(loadMapName('left_skew','med_cluster','ap')),1); ...
                   ];
plotdata = lsk_med./max(lsk_med,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/Med');

subplot(4,3,8);
imagesc(squeeze(mean(interDepResultsMap('left_skew','med_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,9);
imagesc(squeeze(mean(interDepResultsMap('left_skew','med_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,10);
lsk_hi = [mean(loadMapName('left_skew','hi_cluster','h_mi'),1); ...
          mean(abs(loadMapName('left_skew','hi_cluster','cim')),1); ...
          mean(abs(loadMapName('left_skew','hi_cluster','knn_1')),1); ...
          mean(abs(loadMapName('left_skew','hi_cluster','knn_6')),1); ...
          mean(abs(loadMapName('left_skew','hi_cluster','knn_20')),1); ...
          mean(abs(loadMapName('left_skew','hi_cluster','ap')),1); ...
                   ];
plotdata = lsk_hi./max(lsk_hi,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/Hi');

subplot(4,3,11);
imagesc(squeeze(mean(interDepResultsMap('left_skew','hi_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,12);
imagesc(squeeze(mean(interDepResultsMap('left_skew','hi_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

figure;
subplot(4,3,1);
rsk_all = [mean(loadMapName('right_skew','all_cluster','h_mi'),1); ...
           mean(abs(loadMapName('right_skew','all_cluster','cim')),1); ...
           mean(abs(loadMapName('right_skew','all_cluster','knn_1')),1); ...
           mean(abs(loadMapName('right_skew','all_cluster','knn_6')),1); ...
           mean(abs(loadMapName('right_skew','all_cluster','knn_20')),1); ...
           mean(abs(loadMapName('right_skew','all_cluster','ap')),1); ...
                   ];
plotdata = rsk_all./max(rsk_all,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/All');

subplot(4,3,2);
imagesc(squeeze(mean(interDepResultsMap('right_skew','all_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,3);
imagesc(squeeze(mean(interDepResultsMap('right_skew','all_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,4);
rsk_lo = [mean(loadMapName('right_skew','lo_cluster','h_mi'),1); ...
          mean(abs(loadMapName('right_skew','lo_cluster','cim')),1); ...
          mean(abs(loadMapName('right_skew','lo_cluster','knn_1')),1); ...
          mean(abs(loadMapName('right_skew','lo_cluster','knn_6')),1); ...
          mean(abs(loadMapName('right_skew','lo_cluster','knn_20')),1); ...
          mean(abs(loadMapName('right_skew','lo_cluster','ap')),1); ...
                   ];
plotdata = rsk_lo./max(rsk_lo,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/Lo');

subplot(4,3,5);
imagesc(squeeze(mean(interDepResultsMap('right_skew','lo_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,6);
imagesc(squeeze(mean(interDepResultsMap('right_skew','lo_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')


subplot(4,3,7);
rsk_med = [mean(loadMapName('right_skew','med_cluster','h_mi'),1); ...
           mean(abs(loadMapName('right_skew','med_cluster','cim')),1); ...
           mean(abs(loadMapName('right_skew','med_cluster','knn_1')),1); ...
           mean(abs(loadMapName('right_skew','med_cluster','knn_6')),1); ...
           mean(abs(loadMapName('right_skew','med_cluster','knn_20')),1); ...
           mean(abs(loadMapName('right_skew','med_cluster','ap')),1); ...
                   ];
plotdata = rsk_med./max(rsk_med,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/Med');

subplot(4,3,8);
imagesc(squeeze(mean(interDepResultsMap('right_skew','med_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,9);
imagesc(squeeze(mean(interDepResultsMap('right_skew','med_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

subplot(4,3,10);
rsk_hi = [mean(loadMapName('right_skew','hi_cluster','h_mi'),1); ...
          mean(abs(loadMapName('right_skew','hi_cluster','cim')),1); ...
          mean(abs(loadMapName('right_skew','hi_cluster','knn_1')),1); ...
          mean(abs(loadMapName('right_skew','hi_cluster','knn_6')),1); ...
          mean(abs(loadMapName('right_skew','hi_cluster','knn_20')),1); ...
          mean(abs(loadMapName('right_skew','hi_cluster','ap')),1); ...
                   ];

plotdata = rsk_hi./max(rsk_hi,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
% rectangle(ax1,'Position', [numIndependentFeatures,ax1.YLim(1),numIndependentFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [numIndependentFeatures+numRedundantFeatures,ax1.YLim(1),numUselessFeatures,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/Hi');

subplot(4,3,11);
imagesc(squeeze(mean(interDepResultsMap('right_skew','hi_cluster','h_mi'),1)));
hold on;
colormap(cmap)
title('H_{MI}')

subplot(4,3,12);
imagesc(squeeze(mean(interDepResultsMap('right_skew','hi_cluster','cim'),1)));
hold on;
colormap(cmap)
title('CIM')

hh = get(subplot(4,3,12),'Position');
cc = colorbar('Position', [hh(1)+hh(3)+.03  hh(2)+.03  0.02  hh(2)+hh(3)*3]);
set(cc,'fontsize',10, 'ytick',[0, 0.25, .5, 0.75, 1], ...
    'yticklabel', {'0.0', '0.25', '0.50', '0.75', '1.0'}, 'linewidth', 0.5);

%% check the redundancy calculations
