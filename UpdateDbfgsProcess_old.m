function [target,gradient] = UpdateDbfgsProcess_old(D)
%Provide the target function and the gradient.
% Load the data set.
global X;
global weight;
global aa; 
global ll;
global bb;
global L;
global G;
global S;

modProb = exp(X * weight);  % size_sam * size_Y
sumProb = sum(modProb, 2);
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
%prediction

D(D==0)=1e-6; 
% target = bb*sum(sum((modProb-D).^2))+aa*sum(sum((D-S*D).^2))-ll*sum(sum(L.*log(D)));
% binary_cross_entropy = -sum(sum(L.*log(D+1e-6)+ (1-L).*log(1-D+1e-6)));
% binary_cross_entropy_gradient = (-L./(D+1e-6))-((1-L)./(1-D+1e-6));
target = bb*sum(sum((modProb-D).^2))+aa*sum(sum((D-S*D).^2))+ll*sum(sum((L-D).^2));
% target = bb*sum(sum((modProb-D).^2))+aa*trace(D'*G*D)+ll*sum(sum((L-D).^2));
gradient=bb*2*(modProb-D) + 2*aa*(eye(size(D,1))-S')*(D-S*D) + 2*ll*(L-D);
% gradient=bb*2*(modProb-D) + 2*aa*G*D + 2*ll*(L-D);



end