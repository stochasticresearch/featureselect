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
numIndependentFeatures = 4;
numRedundantFeatures = 10;
numUselessFeatures = 36;
skews = {'left_skew','no_skew','right_skew'};
dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
numSamps = 100;

% create redundant feature possibilities
fpCell = {}; fpCellIdx = 1;
for ii=2:numIndependentFeatures
    cnkOut = combnk(1:numIndependentFeatures,ii);
    for jj=1:size(cnkOut,1)
        fpCell{fpCellIdx} = cnkOut(jj,:);
        fpCellIdx = fpCellIdx + 1;
    end
end
operators = {@plus,@times};

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
numMCSims = 10;

% setup output filename
outputFname = sprintf('res_%d_%d_%d_%d_%d.mat',...
    numIndependentFeatures,numRedundantFeatures,numUselessFeatures,numSamps,numMCSims)

% setup estimators and feature selection framework
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
%                        @vmeMI_interface;
                       @apMI_interface;
                       @h_mi_interface};

functionArgsCell    = {{0,1,0};
                       {};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
%                        {};
                       {};
                       {1}};
% fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap','h_mi'};
fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','ap','h_mi'};
numFeaturesToSelect = min(50,numUselessFeatures+numRedundantFeatures+numIndependentFeatures);  % maximum # of features to select

% setup data structure to hold results
resultsMap = containers.Map;
for mk=skews
    zz = containers.Map;
    for sk=dep_clusters
        zzz = containers.Map;
        for f=fNames
            zzz(f) = nan(numMCSims,numFeaturesToSelect);
        end
        zz(sk) = zzz;
    end
    resultsMap(mk) = zz;
end

ispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');

fpCellIdx = 1; operatorIdx = 1; randomFeatureIdx = 1;
ovpIdx = ovpIdx + 1;
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
            corrVec = linspace(0.1,0.4,numIndependentFeatures);
        elseif(strcmp(dc,'med_cluster'))
            corrVec = linspace(0.3,0.7,numIndependentFeatures);
        elseif(strcmp(dc,'hi_cluster'))
            corrVec = linspace(0.6,0.9,numIndependentFeatures);
        else
            corrVec = linspace(0.1,0.9,numIndependentFeatures);
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
                    X(ii,jj) = noSkewContinuousDistInfo.icdf(U(ii,jj));
                end
            end
            % assign output
            y = icdf(noSkewDiscreteDistInfo,U(:,end));
            
            % create redundant features
            curCol = numIndependentFeatures+1;
            for ii=1:numRedundantFeatures
                % get the operator
                op = operators{operatorIdx};
                operatorIdx = mod(operatorIdx,length(operators)) + 1;
                
                % get the operands
                operands = fpCell{fpCellIdx};
                fpCellIdx = mod(fpCellIdx,length(fpCell)) + 1;
                
                % combine & store in XX vector
                res = X(:,operands(1));
                for kk=2:length(operands)
                    res = op(res,XX(:,operands(kk)));
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
                ovp = ovpIdx/(length(skews)*length(dep_clusters)*length(numUselessFeatures));
                dispstat(sprintf('%s--%s OverallProgress=%0.02f',sk, dc, ovp*100));

                % load the results-map from file if it already exists
                if(exist(fullfile(folder,outputFname),'file'))
                    load(fullfile(folder,outputFname));
                end
                fv = resultsMap(sk)(dc)(fNames{fIdx})(mcSimNum,:);
                if(isnan(fv(1)))  % only run the feature selection if we need to!
                    featureVec = mrmr_mid(X, y, numFeaturesToSelect, functionHandlesCell{fIdx}, functionArgsCell{fIdx});
                    % store the results in the appropriate map;
                    resultsMap(sk)(dc)(fNames{fIdx})(mcSimNum,:) = featureVec;

                    % save as we go through the data so that we can pick up where we left off
                    % save only when we update
                    save(fullfile(folder,outputFname),'resultsMap');
                end
                ovpIdx = ovpIdx + 1;
            end
        end
    end
end

%% Generate the scores
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

numIndependentFeatures = 4;
numRedundantFeatures = 10;
numUselessFeatures = 36;
skews = {'left_skew','no_skew','right_skew'};
dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','ap','h_mi'};
numSamps = 100;
numMCSims = 10;

% setup output filename
inputFname = sprintf('res_%d_%d_%d_%d_%d.mat',...
    numIndependentFeatures,numRedundantFeatures,numUselessFeatures,numSamps,numMCSims)
load(fullfile(folder,inputFname))

for sk=skews
    for dc=dep_clusters
        fprintf('***** %s-%s *****',sk,dc);
        for estimator=fNames
            % get the selected matrix
            X = resultsMap(sk)(dc)(estimator);
            score_vec = score_synthetic_fs(X,numIndependentFeatures,numRedundantFeatures,numUselessFeatures);
            fprintf('\t %s-->[%0.02f,%0.02f]',estimator,mean(score_vec),var(score_vec));
        end
    end
end