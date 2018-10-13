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

% manually generate a left-skewed and right-skewed data, from which we
% construct an empirical cdf
leftSkewData = pearsrnd(0,1,-1,3,5000,1);
rightSkewData = pearsrnd(0,1,1,3,5000,1);
[fLeftSkew,xiLeftSkew] = emppdf(leftSkewData,0);
FLeftSkew = empcdf(xiLeftSkew,0);
[fRightSkew,xiRightSkew] = emppdf(rightSkewData,0);
FRightSkew = empcdf(xiRightSkew,0);

% distributions which make up the marginal distributions
leftSkewContinuousDistInfo = rvEmpiricalInfo(xiLeftSkew,fLeftSkew,FLeftSkew,0);
rightSkewContinuousDistInfo = rvEmpiricalInfo(xiRightSkew,fRightSkew,FRightSkew,0);
noSkewContinuousDistInfo = makedist('Normal');
leftSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.1,0.9]);
noSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.5,0.5]);
rightSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.9,0.1]);

% configure the data generation
numIndependentFeatures = 20;
numRedundantFeatures = 20;
numUselessFeatures = 160;
skews = {'left_skew','no_skew','right_skew'};
dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
numSamps = 250;  % run for 250,500

% create redundant feature possibilities
cnkOut = combnk(1:numIndependentFeatures,2);  % only pairwise operations
fpCell = cell(1,length(cnkOut));
for jj=1:size(cnkOut,1)
    fpCell{jj} = cnkOut(jj,:);
end
% operators = {@plus,@times};
operators = {@plus};

% create possibilities for random features
numPossibleRandomFeatures = 10;
randomFeaturesCell = cell(1,numPossibleRandomFeatures);
randomFeaturesCell{1} = makedist('Gamma');
randomFeaturesCell{2} = makedist('Beta');
randomFeaturesCell{3} = makedist('Exponential');
randomFeaturesCell{4} = makedist('ExtremeValue');
randomFeaturesCell{5} = makedist('HalfNormal');
randomFeaturesCell{6} = makedist('InverseGaussian');
randomFeaturesCell{7} = makedist('LogNormal');
randomFeaturesCell{8} = makedist('Rician');
randomFeaturesCell{9} = makedist('Uniform');
randomFeaturesCell{10} = makedist('Weibull');

% setup monte-carlo simulation configuration
numMCSims = 50;

% setup output filename
outputFname = sprintf('res_%d_%d_%d_%d_%d_plusOpOnly.mat',...
    numIndependentFeatures,numRedundantFeatures,numUselessFeatures,numSamps,numMCSims);

% setup estimators and feature selection framework
knn_1 = 1;
knn_6 = 6;
knn_20 = 20;
msi = 0.015625; alpha = 0.2; 
autoDetectHybrid = 0; isHybrid = 1; continuousRvIndicator = 0;

functionHandlesCell = {@taukl_cc_mi_mex_interface;
                       @cim;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @apMI_interface;
                       @h_mi_interface;
                        };
functionArgsCell    = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {1};
                       };
fNames = {'taukl','cim','knn_1','knn_6','knn_20','ap','h_mi'};

numFeaturesToSelect = min(50,numRedundantFeatures+numIndependentFeatures);  % maximum # of features to select

% setup data structure to hold results
selectedFeaturesResultsMap = MapNested();
depWithOutputResultsMap = MapNested();
interDepResultsMap = MapNested();
X_dim_total = numIndependentFeatures+numRedundantFeatures+numUselessFeatures;
numTotalFeatures = numIndependentFeatures+numRedundantFeatures;
for mkIdx=1:length(skews)
    sk = skews{mkIdx};
    for dcIdx=1:length(dep_clusters)
        dc = dep_clusters{dcIdx};
        for fIdx=1:length(fNames)
            f = fNames{fIdx};
            selectedFeaturesResultsMap(sk,dc,f) = nan(numMCSims,numFeaturesToSelect);
            depWithOutputResultsMap(sk,dc,f) = nan(numMCSims,X_dim_total);
            interDepResultsMap(sk,dc,f) = nan(numMCSims,numTotalFeatures,numTotalFeatures);
        end
    end
end

dispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');

operatorIdx = 1; randomFeatureIdx = 1;
ovpIdx = 1;
for skIdx=1:length(skews)
    sk = skews{skIdx};
    if(strcmp(sk,'no_skew'))
        cDistObj = noSkewContinuousDistInfo;
        dDistObj = noSkewDiscreteDistInfo;
    elseif(strcmp(sk,'left_skew'))
        cDistObj = leftSkewContinuousDistInfo;
        dDistObj = leftSkewDiscreteDistInfo;
    else
        cDistObj = rightSkewContinuousDistInfo;
        dDistObj = rightSkewDiscreteDistInfo;
    end
    for dcIdx=1:length(dep_clusters)
        dc = dep_clusters{dcIdx};
        if(strcmp(dc,'lo_cluster'))
            corrVec = linspace(0.15,0.4,numIndependentFeatures);
        elseif(strcmp(dc,'med_cluster'))
            corrVec = linspace(0.3,0.7,numIndependentFeatures);
        elseif(strcmp(dc,'hi_cluster'))
            corrVec = linspace(0.6,0.85,numIndependentFeatures);
        else
            corrVec = linspace(0.15,0.85,numIndependentFeatures);
        end
        
        R = eye(numIndependentFeatures+1);
        R(numIndependentFeatures+1,1:numIndependentFeatures) = corrVec;
        R(1:numIndependentFeatures,numIndependentFeatures+1) = corrVec;
        S = nearestSPD(R);
        R = corrcov(S);

        for mcSimNum=1:numMCSims
            % GENERATE THE DATA
            U = copularnd('Gaussian',R,numSamps);
            X = zeros(numSamps,numIndependentFeatures+numRedundantFeatures+numUselessFeatures);
            
            % assign marginal distributions
            for ii=1:numSamps
                for jj=1:numIndependentFeatures
                    X(ii,jj) = cDistObj.icdf(U(ii,jj));
                end
            end
            % assign output
            y = icdf(dDistObj,U(:,end));
            y(y==1) = -1; y(y==2) = 1;
            
            % create redundant features
            fpCellIdxVec = randsample(1:length(fpCell),numRedundantFeatures);
            curCol = numIndependentFeatures+1;
            for ii=1:numRedundantFeatures
                % get the operator
                op = operators{operatorIdx};
                operatorIdx = mod(operatorIdx,length(operators)) + 1;
                
                % get the operands
                operands = fpCell{fpCellIdxVec(ii)};
                
                % combine & store in XX vector
                res = X(:,operands(1));
                for kk=2:length(operands)
                    res = op(res,X(:,operands(kk)));
                end
                X(:,curCol) = res;
                curCol = curCol + 1;
            end
            
            % create random features & store in X matrix
            for ii=1:numUselessFeatures
                distObj = randomFeaturesCell{randomFeatureIdx};
                X(:,curCol) = random(distObj,numSamps,1);
                curCol = curCol + 1;
                randomFeatureIdx = mod(randomFeatureIdx,numPossibleRandomFeatures)+1;
            end

            % run feature-selection for each algorithm
            for fIdx=1:length(fNames)
                ovp = ovpIdx/(length(skews)*length(dep_clusters)*numMCSims*length(fNames));
                dispstat(sprintf('%s--%s OverallProgress=%0.02f',sk, dc, ovp*100),'timestamp');

                % load the results-map from file if it already exists
                if(exist(fullfile(folder,outputFname),'file'))
                    load(fullfile(folder,outputFname));
                end
                f = fNames{fIdx};
                if(isempty(find(ismember(selectedFeaturesResultsMap(sk,dc).keys,f), 1)))
                    selectedFeaturesResultsMap(sk,dc,f) = nan(numMCSims,numFeaturesToSelect);
                    depWithOutputResultsMap(sk,dc,f) = nan(numMCSims,X_dim_total);
                    interDepResultsMap(sk,dc,f) = nan(numMCSims,X_dim_total);
                end    
                fv_Matrix = selectedFeaturesResultsMap(sk,dc,f);
                featureVec = fv_Matrix(mcSimNum,:);
                if(isnan(featureVec(1)))  % only run the feature selection if we need to!
                    functionHandle = functionHandlesCell{fIdx};
                    argsCell = functionArgsCell{fIdx};
                    
                    % run feature selection
                    featureVec = mrmr_mid(X, y, numFeaturesToSelect, functionHandle, argsCell);
                    % store the results in the appropriate map;
                    fv_Matrix(mcSimNum,:) = featureVec;
                    selectedFeaturesResultsMap(sk,dc,f) = fv_Matrix;
                    
                    % compute raw association between input/output
                    pairwiseVec = zeros(1,X_dim_total);
                    parfor zz=1:X_dim_total
                        pairwiseVec(zz) = functionHandle(X(:,zz),y, argsCell{:});
                    end
                    outputDep_Matrix = depWithOutputResultsMap(sk,dc,f);
                    outputDep_Matrix(mcSimNum,:) = pairwiseVec;
                    depWithOutputResultsMap(sk,dc,f) = outputDep_Matrix;
                    
                    % compute interdependent associations
                    RR = zeros(numTotalFeatures,numTotalFeatures);
                    for zz1=1:numTotalFeatures
                        xx = X(:,zz1);
                        parfor zz2=zz1+1:numTotalFeatures
                            yy = X(:,zz2);
                            RR(zz1,zz2) = functionHandle(xx,yy,argsCell{:});
                        end
                    end
                    RR = RR+RR';  % make it a symmetric matrix by assigning lower triangle to the upper triangle
                    RR(1:numTotalFeatures+1:numTotalFeatures*numTotalFeatures) = 0; % set diagnonal to 0
                    % assign to output
                    interDepTensor = interDepResultsMap(sk,dc,f);
                    interDepTensor(mcSimNum,:,:) = RR;
                    interDepResultsMap(sk,dc,f) = interDepTensor;

                    % save as we go through the data so that we can pick up where we left off
                    % save only when we update
                    save(fullfile(folder,outputFname),...
                        'selectedFeaturesResultsMap',...
                        'depWithOutputResultsMap', ...
                        'interDepTensor');
                end
                ovpIdx = ovpIdx + 1;
            end
        end
    end
end

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
numRedundantFeatures = 20;
numUselessFeatures = 160;
skews = {'left_skew','no_skew','right_skew'};
dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
fNames = {'taukl','cim','knn_1','knn_6','knn_20','ap','h_mi'};
numSamps = 250;
numMCSims = 50;

% setup output filename
inputFname = sprintf('res_%d_%d_%d_%d_%d.mat',...
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
    bw_title = '';
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
%     subplot(1,3,skIdx);
    if(skIdx==length(skews))
        bw_legend = bw_legend_val;
    else
        bw_legend = [];
    end
	figure;
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
% numIndependentFeatures = 20; numRedundantFeatures = 20; 
% numUselessFeatures = 160; numMCSims = 50;
% outputFname = sprintf('res_%d_%d_%d_%d_%d.mat',...
%     numIndependentFeatures,numRedundantFeatures,numUselessFeatures,numSamps,numMCSims);
% loadMapName = depWithOutputResultsMap;
outputFname = 'OpOnly.mat';
load(fullfile(folder,outputFname));
loadMapName = resultsMap;

% create a matrix to compare each of the methods over all the MC
% simulations
cmap = helper_power_colormap();

fillVal = 0;
filler = fillVal*ones(1,200);
yLabelLegend = {'H_{MI}','CIM','KNN-1','KNN-6','KNN-20','AP'};

subplot(4,3,1);
nsk_all = [mean(resultsMap('no_skew','all_cluster','h_mi'),1); ...
           mean(abs(resultsMap('no_skew','all_cluster','cim')),1); ...
           mean(abs(resultsMap('no_skew','all_cluster','knn_1')),1); ...
           mean(abs(resultsMap('no_skew','all_cluster','knn_6')),1); ...
           mean(abs(resultsMap('no_skew','all_cluster','knn_20')),1); ...
           mean(abs(resultsMap('no_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = nsk_all;
plotdata = nsk_all./max(nsk_all,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
hold on;
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/All');

subplot(4,3,2);
lsk_all = [mean(resultsMap('left_skew','all_cluster','h_mi'),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','cim')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','knn_1')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','knn_6')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','knn_20')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = lsk_all;
plotdata = lsk_all./max(lsk_all,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/All');

subplot(4,3,3);
rsk_all = [mean(resultsMap('right_skew','all_cluster','h_mi'),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','cim')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','knn_1')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','knn_6')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','knn_20')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = rsk_all;
plotdata = rsk_all./max(rsk_all,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/All');

subplot(4,3,4);
nsk_lo = [mean(resultsMap('no_skew','lo_cluster','h_mi'),1); ...
          mean(abs(resultsMap('no_skew','lo_cluster','cim')),1); ...
          mean(abs(resultsMap('no_skew','lo_cluster','knn_1')),1); ...
          mean(abs(resultsMap('no_skew','lo_cluster','knn_6')),1); ...
          mean(abs(resultsMap('no_skew','lo_cluster','knn_20')),1); ...
          mean(abs(resultsMap('no_skew','lo_cluster','ap')),1); ...
                   ];
% plotdata = nsk_lo;
plotdata = nsk_lo./max(nsk_lo,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/Lo');

subplot(4,3,5);
lsk_lo = [mean(resultsMap('left_skew','lo_cluster','h_mi'),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','cim')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','knn_1')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','knn_6')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','knn_20')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = lsk_lo;
plotdata = lsk_lo./max(lsk_lo,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/Lo');

subplot(4,3,6);
rsk_lo = [mean(resultsMap('right_skew','lo_cluster','h_mi'),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','cim')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','knn_1')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','knn_6')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','knn_20')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = rsk_lo;
plotdata = rsk_lo./max(rsk_lo,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/Lo');
               
subplot(4,3,7);
nsk_med = [mean(resultsMap('no_skew','med_cluster','h_mi'),1); ...
           mean(abs(resultsMap('no_skew','med_cluster','cim')),1); ...
           mean(abs(resultsMap('no_skew','med_cluster','knn_1')),1); ...
           mean(abs(resultsMap('no_skew','med_cluster','knn_6')),1); ...
           mean(abs(resultsMap('no_skew','med_cluster','knn_20')),1); ...
           mean(abs(resultsMap('no_skew','med_cluster','ap')),1); ...
                   ];
% plotdata = nsk_med;
plotdata = nsk_med./max(nsk_med,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/Med');

subplot(4,3,8);
lsk_med = [mean(resultsMap('left_skew','med_cluster','h_mi'),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','cim')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','knn_1')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','knn_6')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','knn_20')),1); ...
           mean(abs(resultsMap('left_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = lsk_med;
plotdata = lsk_med./max(lsk_med,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/Med');

subplot(4,3,9);
rsk_med = [mean(resultsMap('right_skew','med_cluster','h_mi'),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','cim')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','knn_1')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','knn_6')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','knn_20')),1); ...
           mean(abs(resultsMap('right_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = rsk_med;
plotdata = rsk_med./max(rsk_med,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/Med');
 
subplot(4,3,10);
nsk_hi = [mean(resultsMap('no_skew','hi_cluster','h_mi'),1); ...
          mean(abs(resultsMap('no_skew','hi_cluster','cim')),1); ...
          mean(abs(resultsMap('no_skew','hi_cluster','knn_1')),1); ...
          mean(abs(resultsMap('no_skew','hi_cluster','knn_6')),1); ...
          mean(abs(resultsMap('no_skew','hi_cluster','knn_20')),1); ...
          mean(abs(resultsMap('no_skew','hi_cluster','ap')),1); ...
                   ];
% plotdata = nsk_hi;
plotdata = nsk_hi./max(nsk_hi,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
set(ax1, 'ytick', 1:length(yLabelLegend), 'yticklabel', yLabelLegend, 'FontSize', 12)
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('No-Skew/Hi');

subplot(4,3,11);
lsk_hi = [mean(resultsMap('left_skew','hi_cluster','h_mi'),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','cim')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','knn_1')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','knn_6')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','knn_20')),1); ...
          mean(abs(resultsMap('left_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = lsk_hi;
plotdata = lsk_hi./max(lsk_hi,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Left-Skew/Hi');

subplot(4,3,12);
rsk_hi = [mean(resultsMap('right_skew','hi_cluster','h_mi'),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','cim')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','knn_1')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','knn_6')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','knn_20')),1); ...
          mean(abs(resultsMap('right_skew','all_cluster','ap')),1); ...
                   ];
% plotdata = rsk_hi;
plotdata = rsk_hi./max(rsk_hi,[],2); plotdata(isnan(plotdata))=fillVal;
imagesc(plotdata);
colormap(cmap)
ax1 = gca;
rectangle(ax1,'Position', [ax1.XLim(1),ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [20,ax1.YLim(1),20,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
rectangle(ax1,'Position', [40,ax1.YLim(1),160,ax1.YLim(2)],'EdgeColor','k','LineWidth',2 );
set(ax1, 'ytick', [], 'yticklabel', {}, 'FontSize', 12);
set(ax1, 'xtick', [], 'xticklabel', {}, 'FontSize', 12);
title('Right-Skew/Hi');

hh = get(subplot(4,3,12),'Position');
cc = colorbar('Position', [hh(1)+hh(3)+.03  hh(2)+.03  0.02  hh(2)+hh(3)*3]);
set(cc,'fontsize',10, 'ytick',[0, 0.25, .5, 0.75, 1], ...
    'yticklabel', {'0.0', '0.25', '0.50', '0.75', '1.0'}, 'linewidth', 0.5);

%% check the redundancy calculations

%% check the pairwise dep matrix ONLY!  This is redone above in a comprehensive simulation

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
outputFname = 'pairwise_synth_results_pairwiseRedundant_plusAndTimesOp.mat';

% manually generate a left-skewed and right-skewed data, from which we
% construct an empirical cdf
leftSkewData = pearsrnd(0,1,-1,3,5000,1);
rightSkewData = pearsrnd(0,1,1,3,5000,1);
[fLeftSkew,xiLeftSkew] = emppdf(leftSkewData,0);
FLeftSkew = empcdf(xiLeftSkew,0);
[fRightSkew,xiRightSkew] = emppdf(rightSkewData,0);
FRightSkew = empcdf(xiRightSkew,0);

% distributions which make up the marginal distributions
leftSkewContinuousDistInfo = rvEmpiricalInfo(xiLeftSkew,fLeftSkew,FLeftSkew,0);
rightSkewContinuousDistInfo = rvEmpiricalInfo(xiRightSkew,fRightSkew,FRightSkew,0);
noSkewContinuousDistInfo = makedist('Normal');
leftSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.1,0.9]);
noSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.5,0.5]);
rightSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.9,0.1]);

% configure the data generation
numIndependentFeatures = 20;
numRedundantFeatures = 20;
numUselessFeatures = 160;
skews = {'left_skew','no_skew','right_skew'};
dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
numSamps = 250;  % run for 250,500

cnkOut = combnk(1:numIndependentFeatures,2);  % only pairwise operations
fpCell = cell(1,length(cnkOut));
for jj=1:size(cnkOut,1)
    fpCell{jj} = cnkOut(jj,:);
end
operators = {@plus,@times};
% operators = {@times};

% create possibilities for random features
numPossibleRandomFeatures = 10;
randomFeaturesCell = cell(1,numPossibleRandomFeatures);
randomFeaturesCell{1} = makedist('Gamma');
randomFeaturesCell{2} = makedist('Beta');
randomFeaturesCell{3} = makedist('Exponential');
randomFeaturesCell{4} = makedist('ExtremeValue');
randomFeaturesCell{5} = makedist('HalfNormal');
randomFeaturesCell{6} = makedist('InverseGaussian');
randomFeaturesCell{7} = makedist('LogNormal');
randomFeaturesCell{8} = makedist('Rician');
randomFeaturesCell{9} = makedist('Uniform');
randomFeaturesCell{10} = makedist('Weibull');

knn_1 = 1;
knn_6 = 6;
knn_20 = 20;
msi = 0.015625; alpha = 0.2; 
autoDetectHybrid = 0; isHybrid = 1; continuousRvIndicator = 0;
mine_c = 15;
mine_alpha = 0.6;
rdc_k = 20;
rdc_s = 1/6;

% setup monte-carlo simulation configuration
numMCSims = 50;

functionHandlesCell = {@h_mi_interface;
                       @taukl_cc_mex_interface;
                       @cim;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @apMI_interface;
                       @dcor;
                       @mine_interface_mic;
                       @corr;
                       @rdc;
                       };
functionArgsCell    = {{1};
                       {autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {mine_alpha,mine_c,'mic_e'};
                       {};
                       {rdc_k, rdc_s};
                       };
fNames = {'h_mi','taukl','cim','knn_1','knn_6','knn_20','ap',...
    'dCor','MIC','corr','RDC'};

% setup data structure to hold results
resultsMap = MapNested();
for mkIdx=1:length(skews)
    sk = skews{mkIdx};
    for dcIdx=1:length(dep_clusters)
        dc = dep_clusters{dcIdx};
        for fIdx=1:length(fNames)
            f = fNames{fIdx};
            resultsMap(sk,dc,f) = nan(numMCSims,200);
        end
    end
end

dispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');

fpCellIdx = 1; operatorIdx = 1; randomFeatureIdx = 1;
ovpIdx = 1;
for skIdx=1:length(skews)
    sk = skews{skIdx};
    if(strcmp(sk,'no_skew'))
        cDistObj = noSkewContinuousDistInfo;
        dDistObj = noSkewDiscreteDistInfo;
    elseif(strcmp(sk,'left_skew'))
        cDistObj = leftSkewContinuousDistInfo;
        dDistObj = leftSkewDiscreteDistInfo;
    else
        cDistObj = rightSkewContinuousDistInfo;
        dDistObj = rightSkewDiscreteDistInfo;
    end
    for dcIdx=1:length(dep_clusters)
        dc = dep_clusters{dcIdx};
        if(strcmp(dc,'lo_cluster'))
            corrVec = linspace(0.15,0.4,numIndependentFeatures);
        elseif(strcmp(dc,'med_cluster'))
            corrVec = linspace(0.3,0.7,numIndependentFeatures);
        elseif(strcmp(dc,'hi_cluster'))
            corrVec = linspace(0.6,0.85,numIndependentFeatures);
        else
            corrVec = linspace(0.15,0.85,numIndependentFeatures);
        end
        
        R = eye(numIndependentFeatures+1);
        R(numIndependentFeatures+1,1:numIndependentFeatures) = corrVec;
        R(1:numIndependentFeatures,numIndependentFeatures+1) = corrVec;
        S = nearestSPD(R);
        R = corrcov(S);

        for mcSimNum=1:numMCSims
            % GENERATE THE DATA
            U = copularnd('Gaussian',R,numSamps);
            X = zeros(numSamps,numIndependentFeatures+numRedundantFeatures+numUselessFeatures);
            
            % assign marginal distributions
            for ii=1:numSamps
                for jj=1:numIndependentFeatures
                    X(ii,jj) = cDistObj.icdf(U(ii,jj));
                end
            end
            % assign output
            y = icdf(dDistObj,U(:,end));
            y(y==1) = -1; y(y==2) = 1;
            
            % create redundant features
            fpCellIdxVec = randsample(1:length(fpCell),numRedundantFeatures);
            curCol = numIndependentFeatures+1;
            for ii=1:numRedundantFeatures
                % get the operator
                op = operators{operatorIdx};
                operatorIdx = mod(operatorIdx,length(operators)) + 1;
                
                % get the operands
                operands = fpCell{fpCellIdxVec(ii)};
                
                % combine & store in XX vector
                res = X(:,operands(1));
                for kk=2:length(operands)
                    res = op(res,X(:,operands(kk)));
                end
                X(:,curCol) = res;
                curCol = curCol + 1;
            end
            
            % create random features & store in X matrix
            for ii=1:numUselessFeatures
                distObj = randomFeaturesCell{randomFeatureIdx};
                X(:,curCol) = random(distObj,numSamps,1);
                curCol = curCol + 1;
                randomFeatureIdx = mod(randomFeatureIdx,numPossibleRandomFeatures)+1;
            end

            % run feature-selection for each algorithm
            for fIdx=1:length(fNames)
                ovp = ovpIdx/(length(skews)*length(dep_clusters)*numMCSims*length(fNames));
                dispstat(sprintf('%s--%s OverallProgress=%0.02f',sk, dc, ovp*100),'timestamp');

                % load the results-map from file if it already exists
                if(exist(fullfile(folder,outputFname),'file'))
                    load(fullfile(folder,outputFname));
                end
                f = fNames{fIdx};
                if(isempty(find(ismember(resultsMap(sk,dc).keys,f), 1)))
                    resultsMap(sk,dc,f) = nan(numMCSims,200);
                end 
                % compute every X w/ y
                fv_Matrix = resultsMap(sk,dc,f);
                pairwiseVec = zeros(1,200);
                functionHandle = functionHandlesCell{fIdx};
                argsCell = functionArgsCell{fIdx};
                parfor zz=1:200
                    pairwiseVec(zz) = functionHandle(X(:,zz),y, argsCell{:});
                end
                fv_Matrix(mcSimNum,:) = pairwiseVec;
                resultsMap(sk,dc,f) = fv_Matrix;
                
                save(fullfile(folder,outputFname),'resultsMap');

                ovpIdx = ovpIdx + 1;
            end
        end
    end
end
