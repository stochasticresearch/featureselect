function [score_vec] = score_synthetic_fs(X,numIndepFeatures,numRedundantFeatures,numUselessFeatures)

indepPenalty = (numIndepFeatures+numRedundantFeatures+numUselessFeatures-1).^4;  % max useful penalty
redundantPenalty = (numRedundantFeatures+numUselessFeatures).^2;  % max redundant penalty

numMCSims = size(X,1);
numSelectedFeatures = size(X,2);
numFeaturesToScoreOn = numIndepFeatures+numRedundantFeatures;

score_vec = zeros(1,numMCSims);
for ii=1:numMCSims
    score = 0;
    for jj=1:numFeaturesToScoreOn  % only score the relevant indices.
        selectedFeature = X(ii,jj);
        if(jj<=numIndepFeatures && selectedFeature<=numFeaturesToScoreOn)
            score = score + (selectedFeature-jj).^4;
        elseif(jj<=numIndepFeatures && selectedFeature>numFeaturesToScoreOn)
            score = score + indepPenalty;
        elseif(jj>numIndepFeatures && selectedFeature<=numFeaturesToScoreOn)
            score = score + (selectedFeature-jj).^2;
        elseif(jj>numIndepFeatures && selectedFeature>numFeaturesToScoreOn)
            score = score + redundantPenalty;
        end
    end
    score_vec(ii) = score;
end

end