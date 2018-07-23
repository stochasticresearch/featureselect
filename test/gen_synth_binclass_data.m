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
skews = {'none','right','left'};
dep_clusters = {'lo','med','hi','all'};
numSamps = 250;

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

fpCellIdx = 1; operatorIdx = 1; randomFeatureIdx = 1;
for skIdx=1:length(skews)
    sk = skews{skIdx};
    if(strcmp(sk,'none'))
        cDistObj = noSkewContinuousDistInfo;
        dDistObj = noSkewDiscreteDistInfo;
    elseif(strcmp(sk,'left'))
        cDistObj = leftSkewContinuousDistInfo;
        dDistObj = leftSkewDiscreteDistInfo;
    else
        cDistObj = rightSkewContinuousDistInfo;
        dDistObj = rightSkewDiscreteDistInfo;
    end
    for dcIdx=1:length(dep_clusters)
        dc = dep_clusters{dcIdx};
        if(strcmp(dc,'lo'))
            corrVec = linspace(0.1,0.4,numIndependentFeatures);
        elseif(strcmp(dc,'med'))
            corrVec = linspace(0.3,0.7,numIndependentFeatures);
        elseif(strcmp(dc,'hi'))
            corrVec = linspace(0.6,0.9,numIndependentFeatures);
        else
            corrVec = linspace(0.1,0.9,numIndependentFeatures);
        end
        
        R = eye(numIndependentFeatures+1);
        R(numIndependentFeatures+1,1:numIndependentFeatures) = corrVec;
        R(1:numIndependentFeatures,numIndependentFeatures+1) = corrVec;
        S = nearestSPD(R);
        R = corrcov(S);
        
        U = copularnd('Gaussian',R,numSamps);
        XX = zeros(numSamps,numIndependentFeatures+numRedundantFeatures+numUselessFeatures+1);
        
        % assign marginal distributions
        for ii=1:numSamps
            for jj=1:numIndependentFeatures
                XX(ii,jj) = noSkewContinuousDistInfo.icdf(U(ii,jj));
            end
        end
        % assign output
        XX(:,end) = icdf(noSkewDiscreteDistInfo,U(:,end));
        
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
            res = XX(:,operands(1));
            for kk=2:length(operands)
                res = op(res,XX(:,operands(kk)));
            end
            X(:,curCol) = res;
            curCol = curCol + 1;
        end
        
        % create random features & store in XX vector
        for ii=1:numUselessFeatures
            distObj = randomFeaturesCell{randomFeatureIdx};
            X(:,curCol) = random(distObj,numSamps,1);
            curCol = curCol + 1;
            randomFeatureIdx = mod(randomFeatureIdx,numPossibleRandomFeatures)+1;
        end
        
        % save to disk
        save(fullfile(folder,sprintf('gaussian_%s_%s_%dd.mat',sk,dc,numIndependentFeatures)),'XX');
    end
end

%% load data and process
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

datasets = {'gaussian_none_lo_4d','gaussian_none_med_4d','gaussian_none_hi_4d','gaussian_none_all_4d', ...
            'gaussian_left_lo_4d','gaussian_left_med_4d','gaussian_left_hi_4d','gaussian_left_all_4d', ...
            'gaussian_right_lo_4d','gaussian_right_med_4d','gaussian_right_hi_4d','gaussian_right_all_4d'};

% setup the estimators of MI
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

for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    
    load(fullfile(folder,strcat(dataset,'.mat')),'XX');
    % now test the feature selection algorithms
    X = XX(:,1:end-1);
    y = XX(:,end);
    numFeaturesToSelect = size(X,2);

    for ii=1:length(fNames)
        dispstat(sprintf('\t> Processing %s--%s',dataset,fNames{ii}),'keepthis', 'timestamp');

        fs_outputFname = strcat(dataset,'_fs_',fNames{ii},'.mat');
        fOut = fullfile(folder,fs_outputFname);
        % if file exists, don't re-do it!
        if(~exist(fOut,'file'))
            tic;
            featureVec = mrmr_mid(X, y, numFeaturesToSelect, functionHandlesCell{ii}, functionArgsCell{ii});
            elapsedTime = toc;
            save(fOut,'featureVec','elapsedTime');
        end
    end

end