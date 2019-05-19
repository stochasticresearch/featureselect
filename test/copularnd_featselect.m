function U = copularnd_featselect(copula_type,corrvec,n)
if ~strcmpi(copula_type,'Clayton')
    error('Specified copula type not supported!');
end

d = length(corrvec)+1;
U = zeros(n,d);
U(:,1) = rand(n,1);
for ii=2:d
    alpha = copulaparam(copula_type, corrvec(ii-1), 'type', 'spearman');
    U_tmp = clayton_conditional_cookjohnson(U(:,1),alpha);
    U(:,ii) = U_tmp(:,2);
end

end