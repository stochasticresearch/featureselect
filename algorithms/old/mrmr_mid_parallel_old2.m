function [fea] = mrmr_mid_parallel_old2(d, f, K, miFunctionHandle, miFunctionArgs)
% function [fea] = mrmr_mid_d(d, f, K)
%
% The MID scheme of minimum redundancy maximal relevance (mRMR) feature selection
% 
% The parameters:
%  d - a N*M matrix, indicating N samples, each having M dimensions. Must be integers.
%  f - a N*1 matrix (vector), indicating the class/category of the N samples. Must be categorical.
%  K - the number of features need to be selected
%
% Note: This version only supports discretized data, thus if you have continuous data in "d", you 
%       will need to discretize them first. This function needs the mutual information computation 
%       toolbox written by the same author, downloadable at the Matlab source codes exchange site. 
%       Also There are multiple newer versions on the Hanchuan Peng's web site 
%       (http://research.janelia.org/peng/proj/mRMR/index.htm).
%
% More information can be found in the following papers.
%
% H. Peng, F. Long, and C. Ding, 
%   "Feature selection based on mutual information: criteria 
%    of max-dependency, max-relevance, and min-redundancy,"
%   IEEE Transactions on Pattern Analysis and Machine Intelligence,
%   Vol. 27, No. 8, pp.1226-1238, 2005. 
%
% C. Ding, and H. Peng, 
%   "Minimum redundancy feature selection from microarray gene 
%    expression data," 
%    Journal of Bioinformatics and Computational Biology,
%   Vol. 3, No. 2, pp.185-205, 2005. 
%
% C. Ding, and H. Peng, 
%   "Minimum redundancy feature selection from microarray gene 
%    expression data," 
%   Proc. 2nd IEEE Computational Systems Bioinformatics Conference (CSB 2003),
%   pp.523-528, Stanford, CA, Aug, 2003.
%  
%
% By Hanchuan Peng (hanchuan.peng@gmail.com)
% April 16, 2003
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modified by Kiran Karra to take advantage of some low-hanging fruit for
% speeding up processing, by parallelizing and vectorizing some operations,
% as well as preallocating vectors.
% Contact: kiran.karra@gmail.com
% Github: https://github.com/kkarrancsu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bdisp=0;

nd = size(d,2);
% nc = size(d,1);

t1=cputime;
t = zeros(1,nd);
parfor i=1:nd
    t(i) = miFunctionHandle(d(:,i), f, miFunctionArgs{:});
end
% fprintf('calculate the marginal dmi costs %5.1fs.\n', cputime-t1);

[tmp, idxs] = sort(-t);
fea_base = idxs(1:K);

fea(1) = idxs(1);

% KMAX = min(1000,nd); %500
KMAX = nd;

idxleft = idxs(2:KMAX);

k=1;
if bdisp==1
fprintf('k=1 cost_time=(N/A) cur_fea=%d #left_cand=%d\n', ...
      fea(k), length(idxleft));
end

mi_array = nan(KMAX,K);
for k=2:K
    t1=cputime;
    ncand = length(idxleft);
    curlastfea = length(fea);
   
    t_mi = zeros(1,ncand); 
    parfor i=1:ncand
        t_mi(i) = miFunctionHandle(d(:,idxleft(i)), f, miFunctionArgs{:}); 
        mi_array(i,curlastfea) = getmultimi(d(:,fea(curlastfea)), d(:,idxleft(i)), miFunctionHandle, miFunctionArgs);
    end
    % reshuffle mi_array to be the order intended
    mi_array(idxleft,curlastfea) = mi_array(1:ncand,curlastfea);
    c_mi = nanmean(mi_array(idxleft,:),2)';
    
    [tmp, fea(k)] = max(t_mi(1:ncand) - c_mi(1:ncand));
    tmpidx = fea(k); fea(k) = idxleft(tmpidx); idxleft(tmpidx) = [];

    if bdisp==1
    fprintf('k=%d cost_time=%5.4f cur_fea=%d #left_cand=%d\n', ...
       k, cputime-t1, fea(k), length(idxleft));
    end
end

return;

%===================================== 
function c = getmultimi(da, dt, miFunctionHandle, miFunctionArgs) 
c = zeros(1,size(da,2));
for i=1:size(da,2)
    c(i) = miFunctionHandle(da(:,i), dt, miFunctionArgs{:});
end