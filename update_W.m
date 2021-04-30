function [W_new] = update_W(trainFeature,D_,W,E,Y,beta,miu)

global X;
global D;
global ee;
global mm;
global yy;
global bb;

bb = beta;
X=trainFeature;
D = D_;
ee=E;
mm = miu;
yy=Y;
[W_new,~] = fminlbfgs(@UpdateWbfgsProcess,W);

end