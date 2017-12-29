function Y=read_labels(filename)
%Y=read_labels(filename)
% Read the labels from an ASCII file.

% Isabelle Guyon -- August 2003 -- isabelle@clopinet.com

Y=[];
fp=fopen(filename, 'r');
if fp>0
    Y=fscanf(fp, '%d\n');
    fclose(fp);
end

