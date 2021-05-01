function [target,gradient] = UpdateDbfgsProcess_old(D)
global X;
global weight;
global aa; 
global ll;
global bb;
global L;
global G;
global S;

modProb = exp(X * weight);  
sumProb = sum(modProb, 2);
modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));


D(D==0)=1e-6; 
target = bb*sum(sum((modProb-D).^2))+aa*sum(sum((D-S*D).^2))+ll*sum(sum((L-D).^2));
gradient=bb*2*(modProb-D) + 2*aa*(eye(size(D,1))-S')*(D-S*D) + 2*ll*(L-D);


end
