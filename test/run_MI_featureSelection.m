%% Setup the NIPS Datasets data using routines provided

clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end
datasets = {'dexter','arcene','gisette','madelon'};

for dataset=datasets
    [y_train, y_valid, y_test, X_train, X_valid, X_test] = ...
        read_nips2003_data(fullfile('/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge',dataset,dataset));
    X_train = full(X_train);
    X_valid = full(X_valid);
    save(fullfile(folder,dataset,'data.mat'), 'X_train', 'X_valid', 'X_test', 'y_train', 'y_valid', 'y_test');
end

seed = 12345;
pc_vec = [0.50, 0.75]; % determines how many samples we'll drop from the pos class when training

for skIdx=1:length(pc_vec)
    pos_class_percentage_of_neg = pc_vec(skIdx);
    for dIdx=1:length(datasets)
        dataset = datasets{dIdx};
        dispstat(sprintf('Processing %s',dataset),'keepthis', 'timestamp');

        load(fullfile(folder,dataset,'data.mat'));
        X_train = double(X_train);
        y_train = double(y_train);

        % delete samples from the positive class to ensure we have the
        % percentages as specified above (only for training data!, not
        % validation!)
        I_neg_class = y_train==-1; I_pos_class = y_train==1;
        y_neg_class = y_train(I_neg_class); y_pos_class = y_train(I_pos_class);
        num_desired_y_pos_class = round(length(y_neg_class)*pos_class_percentage_of_neg);
        rng(seed);
        idxs_all = randperm(length(y_pos_class));
        idxs_new_pos_class = idxs_all(1:num_desired_y_pos_class);
        X_neg_class = X_train(I_neg_class,:);
        X_pos_class = X_train(I_pos_class,:); 
        X_pos_class_subsampled = X_pos_class(idxs_new_pos_class,:); 
        y_pos_class_subsampled = y_pos_class(idxs_new_pos_class);

        X_train = [X_neg_class; X_pos_class_subsampled];
        y_train = [y_neg_class; y_pos_class_subsampled];

        save(fullfile(folder,dataset,sprintf('data_skew_%0.02f.mat',pos_class_percentage_of_neg)),...
            'X_train','y_train','X_valid','y_valid');
    end
end

%% Test the mRMR algorithm on various estimators of MI for different datasets
clear;
clc;
dbstop if error;

% setup the estimators of MI
knn_1 = 1; knn_6 = 6; knn_20 = 20;
msi = 0.015625; alpha = 0.2; 
mine_c = 15; mine_alpha = 0.6;
rdc_k = 20; rdc_s = 1/6;

functionHandlesCell = {@taukl_cc_mi_mex_interface;
                       @tau_mi_interface;
                       @cim;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @vmeMI_interface;
                       @apMI_interface;
                       @h_mi_interface;
                       @dcor;
                       @mine_interface_mic;
                       @corr;
                       @rdc;};
autoDetectHybrid = 0; isHybrid = 1; continuousRvIndicator = 0;
functionArgsWithOutputCell    = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {1};
                       {};
                       {mine_alpha,mine_c,'mic_e'};
                       {};
                       {rdc_k, rdc_s};};
autoDetectHybrid = 0; isHybrid = 0; continuousRvIndicator = 0;
functionArgsInterDepCell    = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {1};
                       {};
                       {mine_alpha,mine_c,'mic_e'};
                       {};
                       {rdc_k, rdc_s};};
fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap','h_mi',...
    'dCor','MIC','corr','RDC'};
algosToRun = {'cim','knn_1','knn_6','knn_20','vme','ap','h_mi'};

datasets = {'dexter','arcene','madelon','gisette'};
pc_cell = {'full', 0.50, 0.75}; % determines how many samples we'll drop from the pos class when training

dispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');
for skIdx=1:length(pc_vec)
    pos_class_percentage_of_neg = pc_cell{skIdx};
    
    for dIdx=1:length(datasets)
        dataset = datasets{dIdx};
        if(ispc)
            folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
        elseif(ismac)
            folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
        else
            folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
        end
        dispstat(sprintf('Processing %s',dataset),'keepthis', 'timestamp');

        if(ischar(pos_class_percentage_of_neg))
            load(fullfile(folder,dataset,'data.mat'));
        else
            load(fullfile(folder,dataset,sprintf('data_skew_%0.02f.mat',pos_class_percentage_of_neg)));
        end
        X = double(X_train);
        y = double(y_train);

        numFeaturesToSelect = 50;
        for ii=1:length(algosToRun)
            accessIdx = find(contains(algosToRun{ii},fNames));
            dispstat(sprintf('\t> Processing %s',fNames{accessIdx}),'keepthis', 'timestamp');
            if(ischar(pos_class_percentage_of_neg))
                fs_outputFname = strcat(dataset,'_fs_',fNames{accessIdx},'.mat');
            else
                fs_outputFname = strcat(dataset,'_skew_',sprintf('%0.02f',pos_class_percentage_of_neg),'_fs_',fNames{accessIdx},'.mat');
            end
            fOut = fullfile(folder,dataset,fs_outputFname);
            % if file exists, don't re-do it!
            if(~exist(fOut,'file'))
                tic;
                featureVec = mrmr_mid(X, y, numFeaturesToSelect, ...
                    functionHandlesCell{accessIdx}, ...
                    functionArgsWithOutputCell{accessIdx}, ...
                    functionArgsInterDepCell{accessIdx});
                elapsedTime = toc;
                save(fOut,'featureVec','elapsedTime');
            end
        end
    end
    
end


%% Get the mRMR algorithm's initial feature rankings for analysis
clear;
clc;
dbstop if error;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

% setup the estimators of MI
knn_1 = 1;
knn_6 = 6;
knn_20 = 20;
msi = 0.015625; alpha = 0.2; 
mine_c = 15;
mine_alpha = 0.6;
rdc_k = 20;
rdc_s = 1/6;

functionHandlesCell = {@taukl_cc_mex_interface;
                       @corr;
                       @cim;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @KraskovMI_cc_mex;
                       @vmeMI_interface;
                       @apMI_interface;
                       @h_mi_interface;
                       @dcor;
                       @mine_interface_mic;
                       @corr;
                       @rdc};
autoDetectHybrid = 0; isHybrid = 1; continuousRvIndicator = 0;
functionArgsWithOutputCell    = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {'type','kendall'};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {1};
                       {};
                       {mine_alpha,mine_c,'mic_e'};
                       {};
                       {rdc_k, rdc_s};};
autoDetectHybrid = 0; isHybrid = 0; continuousRvIndicator = 0;
functionArgsInterDepCell    = {{autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {'type','kendall'};
                       {msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator};
                       {knn_1};
                       {knn_6};
                       {knn_20};
                       {};
                       {};
                       {1};
                       {};
                       {mine_alpha,mine_c,'mic_e'};
                       {};
                       {rdc_k, rdc_s};};

fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap','h_mi',...
          'dCor','MIC','corr','RDC'};
algosToRun = {'cim','knn_1','knn_6','knn_20','vme','ap','h_mi'};
      
% datasets = {'dexter','dorothea','gisette','arcene','madelon'};
datasets = {'dexter','gisette','arcene','madelon'};

dispstat('','init'); % One time only initialization
dispstat(sprintf('Begining the simulation...\n'),'keepthis','timestamp');

for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    
    dispstat(sprintf('Processing %s',dataset),'keepthis', 'timestamp');

    load(fullfile(folder,dataset,'data.mat'));
    X = double(X_train);
    y = double(y_train);

    for ii=1:length(algosToRun)
        accessIdx = find(contains(algosToRun{ii},fNames));
        fs_outputFname = strcat(dataset,'_ifs_',fNames{accessIdx},'.mat');
        fOut = fullfile(folder,dataset,fs_outputFname);
        dispstat(sprintf('\t> Processing %s',fNames{accessIdx}),'keepthis', 'timestamp');
        % if file exists, don't re-do it!
        if(~exist(fOut,'file'))
            tic;
            t = mrmr_init_feature_ranking(X, y, ...
                functionHandlesCell{accessIdx}, ...
                functionArgsWithOutputCell{accessIdx}, ...
                functionArgsInterDepCell{accessIdx});
            elapsedTime = toc;
            save(fOut,'t','elapsedTime');
        end
    end
end

%% Analyze the feature selection results for selected dataset

clear;
clc;
dbstop if error;

msi = 0.015625; alpha = 0.2; 
autoDetectHybrid = 0; isHybrid = 1; continuousRvIndicator = 0;

if(ispc)
    folder = 'C:\\Users\\Kiran\\ownCloud\\PhD\\sim_results\\feature_select_challenge';
elseif(ismac)
    folder = '/Users/Kiran/ownCloud/PhD/sim_results/feature_select_challenge';
else
    folder = '/home/kiran/ownCloud/PhD/sim_results/feature_select_challenge';
end

% fNames = {'taukl','tau','cim','knn_1','knn_6','knn_20','vme','ap'};
% fNames = {'cim','knn_1','knn_6','knn_20','vme','ap','h_mi'};
% fNames = {'cim','h_mi','knn_1','knn_6','knn_20','ap'};
fNames = {'cim','h_mi','knn_6','ap'};
titles = {'CIM','H_{MI}','KNN-6','AP'};
datasets = {'dexter','gisette','arcene','madelon'};
skew_levels = {'0.50','0.75','no-skew'};

% produce histogram of the associations of all the features against the
% output matrix, before feature selection for each estimator we care about
nbins = 50;

% compute the overlap of the selected features for each skew-level
num_features_to_compute_ovlp = 50;
for dIdx=1:length(datasets)
    dataset = datasets{dIdx};
    
    load(fullfile(folder,dataset,'data.mat'));
    X = double(X_train);
    y = double(y_train);
    
%     figure;
    for ii=1:length(fNames)
        estimator_name = fNames{ii};
        
        % get the IFS vector which contains the strengths of associations
        % of all features against the output
        fs_outputFname = strcat(dataset,'_ifs_',estimator_name,'.mat');
        fIn = fullfile(folder,dataset,fs_outputFname);
        clear t
        load(fIn);
%         subplot(2,length(fNames),ii);
        figure;
        histogram(t,nbins,'normalization','probability');
%         title(strcat(titles{ii},'-ifs'));
        title(strcat(titles{ii}));
        grid on;
        
%         legendCell = cell(1,length(skew_levels));
%         subplot(2,length(fNames),ii+length(fNames));
%         estimator_selected_features = zeros(3,num_features_to_compute_ovlp);  % 3 skews, 50 max features selected
%         for jj=1:length(skew_levels)
%             skl = skew_levels{jj};
%             if(strcmp(skl,'no-skew'))
%                 fs_Fname = strcat(dataset,'_fs_',estimator_name,'.mat');
%             else
%                 fs_Fname = strcat(dataset,'_skew_',skl,'_fs_',estimator_name,'.mat');
%             end
%             fIn = fullfile(folder,dataset,fs_Fname);
%             % remove previous result for safety
%             clear featureVec
%             if(exist(fIn,'file'))
%                 load(fIn);
%                 estimator_selected_features(jj,:) = featureVec(1:num_features_to_compute_ovlp);
%             end
%             featureVec_strength_of_association = t(featureVec);
%             histogram(featureVec_strength_of_association,nbins,'normalization','probability');
%             legendCell{jj} = skl;
%             hold on;
%         end
%         title(fNames{ii});
%         legend(legendCell);

%         % for the selected features, compute the monotonicity using CIM
%         feat_sel_vec = estimator_selected_features(3,:);
%         num_regions_arr = zeros(1,size(feat_sel_vec,2));
%         for kk=1:size(feat_sel_vec,2)
%             xxx = X(:,feat_sel_vec(kk));
%             [~,regions] = cim(xxx,y,msi,alpha,autoDetectHybrid,isHybrid,continuousRvIndicator);
%             num_regions = size(regions,2);
%             num_regions_arr(kk) = num_regions;
%         end
        
        % compute the % of features which remained the same and output that
        % information
%         ovlp_noskew_and_fifty = length(intersect(estimator_selected_features(3,:),estimator_selected_features(1,:)))*100/num_features_to_compute_ovlp;
%         ovlp_noskew_and_seventyfive = length(intersect(estimator_selected_features(3,:),estimator_selected_features(2,:)))*100/num_features_to_compute_ovlp;
%         ovlp_fifty_and_seventyfive = length(intersect(estimator_selected_features(2,:),estimator_selected_features(1,:)))*100/num_features_to_compute_ovlp;
%         ovlp_all = length(intersect(estimator_selected_features(3,:),intersect(estimator_selected_features(2,:),estimator_selected_features(1,:))))*100/num_features_to_compute_ovlp;
%         fprintf('%s--%s 1.0/0.5=%0.02f 1.0/0.75=%0.02f, 0.5/0.75=%0.02f, all=%0.02f\n',...
%             dataset,estimator_name,ovlp_noskew_and_fifty,...
%             ovlp_noskew_and_seventyfive,ovlp_fifty_and_seventyfive,ovlp_all);
%         fprintf('%s--%s all=%0.02f\n',...
%             dataset,estimator_name,ovlp_all);
%         fprintf('%s--%s 1.0/0.5=%0.02f 1.0/0.75=%0.02f mean(#regions)=%0.02f \n',...
%             dataset,estimator_name,ovlp_noskew_and_fifty,...
%             ovlp_noskew_and_seventyfive,mean(num_regions_arr));
        
    end
%     suptitle(upper(dataset));
%     fprintf('\n');
end

%% some tests to make sure that the serial and parallel versions of the

% % algorithm produce the same results
% testIdx = 1;
% K = 30;
% tic;
% 
% fea1 = mrmr_mid_serial(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
% fea2 = mrmr_mid(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
% % fea3 = mrmr_mid_parallel_old2(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
% % fprintf('1==2?=%d 1==3?=%d\n',isequal(fea1,fea2), isequal(fea1,fea3));
% fprintf('1==2?=%d\n',isequal(fea1,fea2));
% 
% % fea1 = mrmr_mid_debug(X, y, K, functionHandlesCell{testIdx}, functionArgsCell{testIdx})
