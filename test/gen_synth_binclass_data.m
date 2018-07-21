clear;
clc;

% Test it out w/ a 5-D Gaussian
R = [ [1.0 0.0 0.0 0.0 0.4]; ...
      [0.0 1.0 0.0 0.0 0.3]; ...
      [0.0 0.0 1.0 0.0 0.2]; ...
      [0.0 0.0 0.0 1.0 0.3]; ...
      [0.4 0.3 0.2 0.3 1.0] ];
numSamps = 1000;
U = copularnd('Gaussian',R,numSamps);
X = zeros(size(U));

% manually generate a left-skewed and right-skewed data, from which we
% construct an empirical cdf
leftSkewData = pearsrnd(0,1,-1,3,5000,1);
rightSkewData = pearsrnd(0,1,1,3,5000,1);
[fLeftSkew,xiLeftSkew] = emppdf(leftSkewData,0);
FLeftSkew = empcdf(xiLeftSkew,0);
[fRightSkew,xiRightSkew] = emppdf(rightSkewData,0);
FRightSkew = empcdf(xiRightSkew,0);

% distributions which make up the marginal distributions
leftSkewContinuousDistInfo = rvEmpiricalInfo(xiLeftSkew,fLeftSkew,FLeftSkew,0);
rightSkewContinuousDistInfo = rvEmpiricalInfo(xiRightSkew,fRightSkew,FRightSkew,0);
noSkewContinuousDistInfo = makedist('Normal');
leftSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.1,0.9]);
noSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.5,0.5]);
rightSkewDiscreteDistInfo = makedist('Multinomial','probabilities',[0.9,0.1]);

% assign marginal distributions
for ii=1:numSamps
    X(ii,1) = noSkewContinuousDistInfo.icdf(U(ii,1));
    X(ii,2) = noSkewContinuousDistInfo.icdf(U(ii,2));
    X(ii,3) = noSkewContinuousDistInfo.icdf(U(ii,3));
    X(ii,4) = noSkewContinuousDistInfo.icdf(U(ii,4));
end
X(:,5) = icdf(leftSkewDiscreteDistInfo,U(:,5));

plotmatrix(X)

% assign the output

% create fake features which are combinations of the 5 independent features
% we created

% zip up the whole dataset into one Matrix
