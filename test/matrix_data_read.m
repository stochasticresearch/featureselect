function X=matrix_data_read(filename, featnum, patnum, data_type)
%X=matrix_data_read(filename, featnum, patnum, data_type)
% Read the a matrix from an ASCII file in the ASCII format
% of the benchmark. The program "guesses" whether it is a
% regular matrix or a sparse matrix by reading the first few lines.
% This is not completely bullet proof, but it works for the
% datasets of interest.
% The sparse matrix format for binary data consists of lines of 
% indices of non-zero elements.
% The sparse matrix format for integer data onsists of lines of 
% indices of non-zero elements followed by a colon followed by the
% values. 
% The non-sparse matrix format is just a regular table.
% The delimeters expected are spaces between values and new lines
% between lines.
% Inputs:
% filename -- Name of the file to read from.
% featnum -- Number of features.
% patnum -- Number of patterns.
% data_type -- Type of data: non-sparse, sparse-integer, or sparse-binary.
% Returns:
% X -- Matrix read.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

fid=fopen(filename, 'r');
fprintf('Reading file %s, please be patient ...\n', filename);

if nargin<3, patnum=0; end
if nargin<4,
    fn=zeros(1,10);
    % Guesses the format
    data_type='non-sparse';
	for i=1:10
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        if findstr(tline, ':')
            data_type='sparse-integer';
        else
            mm=sscanf(tline,'%d ');
            fn(i)=length(mm);
        end
        patnum=patnum+1;  
	end
	if any(fn~=featnum) & ~strcmp(data_type, 'sparse-integer')
        data_type='sparse-binary';
	end
end
if nargin<3,
    % Counts the patterns
    while 1
        tline = fgetl(fid);
        if ~ischar(tline), break, end
        patnum=patnum+1;
    end
end
fclose(fid);
fprintf('Data type: %s \n', data_type);
fprintf('Pattern number: %d \n', patnum);
fprintf('Feature number: %d \n', featnum);
fprintf('Percent done: ');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fid=fopen(filename, 'r');
k=1;
percent_done=0;
old_percent_done=0;
if strcmp(data_type, 'non-sparse')
    X=zeros(patnum, featnum);
else
    X=sparse(patnum, featnum);
end    
while 1
    percent_done=floor(k/patnum*100);
    if ~mod(percent_done,10) & percent_done~=old_percent_done,
        fprintf('%d%% ', percent_done);
    end
    old_percent_done=percent_done;
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    switch data_type
        case 'sparse-binary'
            idx=sscanf(tline,'%d ');
            val=ones(1,length(idx));
            X(k,:)=sparse(1,idx',val',1,featnum);
        case 'sparse-integer'
            m=sscanf(tline,'%d:%g ');
            idx=m(1:2:length(m));
            val=m(2:2:length(m));
            X(k,:)=sparse(1,idx',val',1,featnum);
        otherwise
            m=sscanf(tline,'%d ');
            X(k,:)=m';
    end
    k=k+1;
end
sparsity=1 - nnz(X)/prod(size(X));
if sparsity>0.85
    X=sparse(X);
end
fprintf(' ... done!\n');
fclose(fid);