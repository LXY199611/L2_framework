function modProb = predictLDL(x,weights)
modProb = exp(x * weights);
sumProb = sum(modProb, 2); 
modProb = scalecols(modProb, 1 ./ sumProb);

function modProb = scalecols(x, s)
[~, numCols] = size(x); 
modProb = x .* repmat(s, 1, numCols);
end
end
