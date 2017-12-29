function [Y_train, Y_valid, Y_test, X_train, X_valid, X_test] = read_nips2003_data(input_name,data_dir,data_name)

p=read_parameters([input_name '.param']);

Y_train=read_labels([input_name '_train.labels']);
Y_valid=read_labels([input_name '_valid.labels']);  
Y_test=read_labels([input_name '_test.labels']);   
% Read the data
X_train=matrix_data_read([input_name '_train.data'],p.feat_num,p.train_num,p.data_type);
X_valid=matrix_data_read([input_name '_valid.data'],p.feat_num,p.valid_num,p.data_type);
X_test=matrix_data_read([input_name '_test.data'],p.feat_num,p.test_num,p.data_type);

end