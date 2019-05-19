clear;
clc;
dbstop if error;
rng(123);

% parpool(6);

if(ispc)
    error('unsupported OS!');
elseif(ismac)
    folder = '/Users/karrak1/Documents/erc_paper';
else
    folder = '/home/apluser/stochasticresearch/data/erc_paper';
end

% distributions which make up the marginal distributions
% see here: https://probabilityandstats.files.wordpress.com/2015/05/uniform-densities.jpg
% for mapping between the skew distributions & beta, or just plot pdf
leftSkewContinuousDistInfo = makedist('Beta', 'a', 20, 'b', 4);
rightSkewContinuousDistInfo = makedist('Beta', 'a', 4, 'b', 20);
noSkewContinuousDistInfo = makedist('Beta', 'a', 10, 'b', 10);
leftSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.1,0.9]);
noSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.5,0.5]);
rightSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.9,0.1]);

% configure the data generation
numIndependentFeatures = 20;
numRedundantFeatures = 20;
numUselessFeatures = 160;
skews = {'left_skew','no_skew','right_skew'};
dep_clusters = {'lo_cluster','med_cluster','hi_cluster','all_cluster'};
numSamps = 100;  % run for 250,500

% create redundant feature possibilities
cnkOut = combnk(1:numIndependentFeatures,2);  % only pairwise operations
fpCell = cell(1,length(cnkOut));
for jj=1:size(cnkOut,1)
    fpCell{jj} = cnkOut(jj,:);
end
% operators = {@plus,@times};
% operators = {@times};
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
numMCSims = 10;
% copula_type='Gaussian';
copula_type = 't';
DoF = 2;

% setup output filename
if(strcmpi(copula_type,'gaussian'))
    outputFname = sprintf('res_%d_%d_%d_%d_%d_%s_plusOpOnly.mat',...
    numIndependentFeatures,numRedundantFeatures,numUselessFeatures,...
    numSamps,numMCSims,copula_type);
elseif(strcmpi(copula_type,'t'))
    outputFname = sprintf('res_%d_%d_%d_%d_%d_%s_%d_plusOpOnly.mat',...
    numIndependentFeatures,numRedundantFeatures,numUselessFeatures,...
    numSamps,numMCSims,copula_type, DoF);
end


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
% functionHandlesCell = {@h_mi_interface;};
functionArgs_withOutput_Cell = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                                {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                                {knn_1};
                                {knn_6};
                                {knn_20};
                                {};
                                {1};
                                };
% functionArgs_withOutput_Cell = {{1};};
isHybrid = 0;  % we compare against each other continuous features, so it is not hybrid
functionArgs_interDep_Cell = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                              {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                              {knn_1};
                              {knn_6};
                              {knn_20};
                              {};
                              {1};
                              };
% functionArgs_interDep_Cell = {{1};};
fNames = {'taukl','cim','knn_1','knn_6','knn_20','ap','h_mi'};
% fNames = {'h_mi'};

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
        R = corrcov(nearcorr(R));

        for mcSimNum=1:numMCSims
            % GENERATE THE DATA
            if(strcmpi(copula_type,'Gaussian'))
                U = copularnd('Gaussian',R,numSamps);
            elseif(strcmpi(copula_type,'t'))
                U = copularnd('t',R,DoF,numSamps);
            else
                U = copularnd_featselect(copula_type,corrVec,numSamps);
            end
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
                    argsWithOutput_Cell = functionArgs_withOutput_Cell{fIdx};
                    argsInterDep_Cell = functionArgs_interDep_Cell{fIdx};
                    
                    % run feature selection
                    [featureVec,pairwiseVec] = mrmr_mid(X, y, numFeaturesToSelect, functionHandle, argsWithOutput_Cell, argsInterDep_Cell);
                    % store the results in the appropriate map;
                    fv_Matrix(mcSimNum,:) = featureVec;
                    selectedFeaturesResultsMap(sk,dc,f) = fv_Matrix;
                    
                    outputDep_Matrix = depWithOutputResultsMap(sk,dc,f);
                    outputDep_Matrix(mcSimNum,:) = pairwiseVec;
                    depWithOutputResultsMap(sk,dc,f) = outputDep_Matrix;
                    
                    % compute interdependent associations
                    RR = zeros(numTotalFeatures,numTotalFeatures);
                    for zz1=1:numTotalFeatures
                        xx = X(:,zz1);
                        parfor zz2=zz1+1:numTotalFeatures
                            yy = X(:,zz2);
%                             RR(zz1,zz2) = functionHandle(xx,yy,argsCell{:});
                            RR(zz1,zz2) = feval(functionHandle,xx,yy,argsInterDep_Cell{:});
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
                        'interDepResultsMap');
                end
                ovpIdx = ovpIdx + 1;
            end
        end
    end
end