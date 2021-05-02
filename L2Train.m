function [D_,W,loss_list]=L2Train(trainFeature,trainLogicalLabel,op)
D_=build_label_manifold(trainFeature,trainLogicalLabel,op.k); 
W=rand(size(trainFeature,2),size(trainLogicalLabel,2));
Y = zeros(size(trainFeature,2),size(trainLogicalLabel,2));
E = W;
[~,d] = size(trainFeature);
max_iter = 200;
convergence2=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);
epsilon_abs=1e-5;
epsilon_rel=1e-5;
loss_list = zeros(max_iter,1);
% miu = 1e-4;
miu = op.miu;
miu_max = 1e6;
rho = 1.1;
        
for i = 1:max_iter
    S = obtain_S(trainFeature,D_,op.k,op.alpha,op.alpha); 
    D_ = update_D_old(trainFeature,trainLogicalLabel,D_,S,W,op.beta,op.alpha,op.lambda);
    D_ = real(D_);
    W = update_W(trainFeature,D_,W,E,Y,op.beta,miu);
    W = real(W);
    E = update_E(W,Y,op.gamma,miu);
    E = real(E);
    Y = Y+miu*(W-E);
    modProb = exp(trainFeature * W); 
    sumProb = sum(modProb, 2);
    modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));    
    convergence2(i,1)=norm(miu*(W-E),'fro'); 
    epsilon_dual(i,1)=sqrt(d)*epsilon_abs+epsilon_rel*norm(Y,'fro');
    if (convergence2(i,1)<=epsilon_dual(i,1))
        break;
    end
    miu = min(miu*rho, miu_max) ;

end
D_ = softmax(D_);
sumD = sum(D_, 2);
D_ = scalecols(D_, 1 ./ sumD);
end
function modProb = scalecols(x, s)
[~, numCols] = size(x); 
modProb = x .* repmat(s, 1, numCols);
end
