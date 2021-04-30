function [new_E] = update_E(W,Y,gamma,miu)
Q=W+Y/miu;

the=gamma/miu;

[new_E,~]=svdThreshold(Q,the);

end