function [dd] = mrmr_init_feature_ranking(d, f, miFunctionHandle, miFunctionArgs, KMAX_in)

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
