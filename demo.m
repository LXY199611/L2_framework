clear;
clc;
load sample;
load parameter;
[D_,W,loss]=L2Train(train_X,train_L,op);
testPrediction = softmax((test_X*W)')';
