function [score_vec] = score_synthetic_fs(X,numIndepFeatures,numRedundantFeatures,numUselessFeatures)

indepPenalty = numIndepFeatures+numRedundantFeatures+numUselessFeatures;
redundantPenalty = numRedundantFeatures+numUselessFeatures;

numMCSims = size(X,1);
numFeaturesToScoreOn = numIndepFeatures+numRedundantFeatures;

score_vec = zeros(1,numMCSims);
for ii=1:numMCSims
    score = 0;
    for jj=1:numFeaturesToScoreOn  % only score the relevant indices.
        selectedFeature = X(ii,jj);
        if(jj<=numIndepFeatures && selectedFeature<=numFeaturesToScoreOn)
            % this scoring mechanism says that higher indexed features are
            % more informative of the output than lower correlated
            % features, which is how we designed the correlation matrix for
            % the Gaussian copula.  For example, if we have 4 independent
            % features, then the feature in the 4th column has a higher
            % correlation coefficient w/ the output than the 3rd column.
            % Look at Lines 131-141 of gen_synth_binclass_mctests.m to
            % understand why.
            score = score + (numIndepFeatures-selectedFeature);
        elseif(jj<=numIndepFeatures && selectedFeature>numFeaturesToScoreOn)
            score = score + indepPenalty;
        elseif(jj>numIndepFeatures && selectedFeature<=numFeaturesToScoreOn)
            % the features that are redundant, but informative, incur a
            % constant penalty because it is hard to say which one is more
            % informative than the other.
            score = score + (numFeaturesToScoreOn-numIndepFeatures);
        elseif(jj>numIndepFeatures && selectedFeature>numFeaturesToScoreOn)
            score = score + redundantPenalty;
        end
    end
    score_vec(ii) = score;
end

end