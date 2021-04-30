function [Zk,traceNorm]=svdThreshold(svd_obj,lambdadL)

% [V,D]=eig(svd_obj'*svd_obj);
% D=diag(D);
% %D(D<1e-10)=0;
% D(D<0)=0;
% D=sqrt(D)';
% D2=D;
% D2(D~=0)=D(D~=0).^(-1);
% D=diag(max(0,D-lambdadL));
% Zk=svd_obj*((V*diag(D2))*(D*V'));
% traceNorm=sum(diag(D));

eig_obj=svd_obj'*svd_obj;
eig_obj2=eig_obj;
eig_tag=0;
eig_obj_timer=1e20;
while (eig_tag==0)&&(eig_obj_timer>=1e8)
    try 
        [V,D]=eig(eig_obj2);
        eig_tag=1;
    catch
        eig_tag=0;
        eig_obj2=round(eig_obj*eig_obj_timer)/eig_obj_timer;
        eig_obj_timer=eig_obj_timer/10;
    end
end

D=diag(D);
%D(D<1e-10)=0;
D(D<0)=0;
D=sqrt(D)';
D2=D;
D2(D~=0)=D(D~=0).^(-1);
D=diag(max(0,D-lambdadL));
Zk=svd_obj*((V*diag(D2))*(D*V'));
traceNorm=sum(diag(D));

end