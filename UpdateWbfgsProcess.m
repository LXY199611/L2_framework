function [target,gradient] = UpdateWbfgsProcess(W)
%Provide the target function and the gradient.
% Load the data set.
global X;
global D;
global ee;
global mm;
global yy;
global bb;

modProb = exp(X * W);  % size_sam * size_Y
    % modProb = X * W;
sumProb = sum(modProb, 2);
W(isnan(W)) = 1e-6;
W(isinf(W)) = 1e10;
sumProb(isnan(sumProb)) = 1e-4;
sumProb(isinf(sumProb)) = 1e10;
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
modProb(isnan(modProb)) = 1e-4;
modProb(isinf(modProb)) = 1e10;

%prediction
target = bb*sum(sum((modProb-D).^2))+ mm/2*sum(sum((W-ee+yy/mm).^2));
target = real(target);
target(isnan(target)) = 0;
% if isinf(target)
%     target = 1e10;
% end
% if isnan(target)
%     target = 1e-4;
% end
gradient = 2*bb*X'*((modProb - D).*(modProb-modProb.*modProb)) + mm*(W-ee+yy/mm) ;
gradient = real(gradient);
gradient(isnan(gradient)) = 0;
end


