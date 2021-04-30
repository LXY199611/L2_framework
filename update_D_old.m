function [D_new] = update_D_old(trainFeature,trainlogicalLabel,D_,Similarity,W,beta,alpha,lambda)
global X;
global weight;
global aa; 
global ll;
global bb;
global L;
global S;
global G;

S=Similarity;
L=trainlogicalLabel;
ll=lambda;
X = trainFeature;
weight = W;
aa=alpha;
bb = beta;
for i=1:size(S,1)
di(i)=0;
    for j=1:size(S,2)
        di(i)=di(i)+(S(i,j)+S(j,i))/2;
    end
end
S_hat = diag(di);
G=S-S_hat;

[D_new,~] = fminlbfgs(@UpdateDbfgsProcess_old,D_);
D_new(D_new<0)=0;
D_ = exp(D_);
D_=D_./repmat(sum(D_ ,2),1,size(D_ ,2));
end