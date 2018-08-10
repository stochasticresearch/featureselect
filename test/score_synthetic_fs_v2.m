function [score_vec] = score_synthetic_fs_v2(X,numIndepFeatures,numRedundantFeatures)

numFeaturesToScoreOn = numIndepFeatures+numRedundantFeatures;

indepFeatureSelectScore = 0;
redundantFeatureSelectScore = 1;
remainderScore = 0;
% in this scoring mechanism, you just get points for every relevant
% feature you select, with a configurable difference between whether you 
% selected a redundant feature or a direct feature

% score_vec = sum(X(:,1:numFeaturesToScoreOn)<=numFeaturesToScoreOn,2);
iiIndependent = X(:,1:numFeaturesToScoreOn)<=numIndepFeatures;
iiRedundant = X(:,1:numFeaturesToScoreOn)>numIndepFeatures & X(:,1:numFeaturesToScoreOn)<=numFeaturesToScoreOn;
iiRemainder = X(:,1:numFeaturesToScoreOn)>=numFeaturesToScoreOn;
X(iiIndependent) = indepFeatureSelectScore;
X(iiRedundant) = redundantFeatureSelectScore;
X(iiRemainder) = remainderScore;

score_vec = sum(X(:,1:numFeaturesToScoreOn),2);

end