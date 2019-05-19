% test the copularnd_featselect.m functionality
copula_type='Clayton';
d = 3;
corrvec = linspace(0.7,0.95,d);
num_samps = 500;
U = copularnd_featselect(copula_type,corrvec,num_samps);
plotmatrix(U)