function [D_,W,loss_list]=L2Train(trainFeature,trainLogicalLabel,op)
D_=build_label_manifold(trainFeature,trainLogicalLabel,op.k); 
%initial recovered Label Distribution
W=rand(size(trainFeature,2),size(trainLogicalLabel,2));
Y = zeros(size(trainFeature,2),size(trainLogicalLabel,2));
E = W;
[~,d] = size(trainFeature);
op.beta=1e-4;
max_iter = 200;
convergence2=zeros(max_iter,1);
epsilon_dual=zeros(max_iter,1);
epsilon_abs=1e-5;
epsilon_rel=1e-5;
loss_list = zeros(max_iter,1);
miu = 1e-4;
miu_max = 1e6;
rho = 1.1;
        
for i = 1:max_iter
    S = obtain_S(trainFeature,D_,op.k,op.alpha,op.alpha); 
%     loss_list(i)= obj_func1(trainFeature,trainLogicalLabel,D_,S,W,op);
    D_ = update_D_old(trainFeature,trainLogicalLabel,D_,S,W,op.beta,op.alpha,op.lambda);
    W = update_W(trainFeature,D_,W,E,Y,op.beta,miu);
    E = update_E(W,Y,op.gamma,miu);
    Y = Y+miu*(W-E);
    modProb = exp(trainFeature * W);  % size_sam * size_Y
    sumProb = sum(modProb, 2);
    modProb = modProb ./ (repmat(sumProb,[1 size(modProb,2)]));
    
    
    %dual residual
    convergence2(i,1)=norm(miu*(W-E),'fro');
  
    
    %dual epsilon
    epsilon_dual(i,1)=sqrt(d)*epsilon_abs+epsilon_rel*norm(Y,'fro');
    
    if (convergence2(i,1)<=epsilon_dual(i,1))
        break;
    end
    % update the Lagrange multiplier Y and regularization term miu
    miu = min(miu*rho, miu_max) ;

end
D_ = softmax(D_);
sumD = sum(D_, 2); % sum of rows
D_ = scalecols(D_, 1 ./ sumD);
end
function modProb = scalecols(x, s)
[~, numCols] = size(x); 
modProb = x .* repmat(s, 1, numCols);
end