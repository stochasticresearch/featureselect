function [fea] = mrmr_mid(d, f, K, miFunctionHandle, miFunctionArgs, KMAX_in)
% The MID scheme of minimum redundancy maximal relevance (mRMR) feature selection
% 
% The parameters:
%  d - a N*M matrix, indicating N samples, each having M dimensions. Must be integers.
%  f - a N*1 matrix (vector), indicating the class/category of the N samples. Must be categorical.
%  K - the number of features need to be selected
%  miFunctionHandle - Function handle to the estimator of mutual
%                     information
%  miFunctionArgs - Cell array of the arguments required for this estimator
%                   of mutual information (after X & Y)
%  KMAX_in - the maximum number of top ranking features to consider when
%            doing feature selection.  The larger teh number, the better
%            the results, but also more computation required
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

nd = size(d,2);
if(nargin<6)
    KMAX = min(1000,nd); % the # of top features to consider when searching 
                     % using the mRMR algorithm
else
    KMAX = min(KMAX_in,nd);
end

t = zeros(1,nd);
parfor i=1:nd
    t(i) = miFunctionHandle(d(:,i), f, miFunctionArgs{:});
end

[~, idxsOriginal] = sort(-t);
dd = d(:,idxsOriginal(1:KMAX));  % hash the data down for efficiency

fea(1) = 1;
idxleft = 2:KMAX;

mi_array = nan(KMAX,K);
for k=2:K
    ncand = length(idxleft);
    curlastfea = length(fea);
   
    t_mi = zeros(1,ncand); 
    parfor i=1:ncand
        t_mi(i) = miFunctionHandle(dd(:,idxleft(i)), f, miFunctionArgs{:}); 
        mi_array(i,curlastfea) = getmultimi(dd(:,fea(curlastfea)), dd(:,idxleft(i)), miFunctionHandle, miFunctionArgs);
    end
    mi_array(idxleft,curlastfea) = mi_array(1:ncand,curlastfea);
    c_mi = nanmean(mi_array(idxleft,:),2)';
    
    [~, fea(k)] = max(t_mi(1:ncand) - c_mi(1:ncand));
    tmpidx = fea(k); fea(k) = idxleft(tmpidx); idxleft(tmpidx) = [];

end

% map the features back to the original features
fea = idxsOriginal(fea);

return;

%===================================== 
function c = getmultimi(da, dt, miFunctionHandle, miFunctionArgs) 
c = zeros(1,size(da,2));
for i=1:size(da,2)
    c(i) = miFunctionHandle(da(:,i), dt, miFunctionArgs{:});
end