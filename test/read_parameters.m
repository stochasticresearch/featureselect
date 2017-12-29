function p=read_parameters(filename)
%p=read_parameters(filename)
% Read dataset parameters and statistics

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

fp=fopen(filename, 'r');

p.data_type=fscanf(fp, 'Data type: %s\n');
p.feat_num=fscanf(fp, 'Number of features: %d\n');
fgetl(fp);
fgetl(fp);
m=fscanf(fp, 'Train\t%5d\t%5d\t%5d\t%g\n');
p.train_pos_num=m(1);
p.train_neg_num=m(2);
p.train_num=m(3);
p.train_check_sum=m(4);
m=fscanf(fp, 'Valid\t%5d\t%5d\t%5d\t%g\n');
p.valid_pos_num=m(1);
p.valid_neg_num=m(2);
p.valid_num=m(3);
p.valid_check_sum=m(4);
m=fscanf(fp, 'Test\t%5d\t%5d\t%5d\t%g\n');
p.test_pos_num=m(1);
p.test_neg_num=m(2);
p.test_num=m(3);
p.test_check_sum=m(4);
m=fscanf(fp, 'All\t%5d\t%5d\t%5d\t%g\n');
p.all_pos_num=m(1);
p.all_neg_num=m(2);
p.all_num=m(3);
p.all_check_sum=m(4);

fclose(fp);
