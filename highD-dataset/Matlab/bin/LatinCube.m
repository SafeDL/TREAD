%% 拉丁超方体抽样
tic 
clc,clear all; 
% X =  bestlh(2000,6,50,10); % 分别是样本个数、维数、Population以及iteration:Forester的书
X = lhsamp(100000,3); % DACE工具箱
toc
%% define a range for starting values
dhwmin = 0;
dhwmax = 150;
XVelocitymin = 0;
XVelocitymax = 50;
rVelocitymin = -25;
rVelocitymax = 0;
preXvelmin = 0.1;
preXvelmax = 50;
preAxmin = -10;
preAxmax = 4;
timemin = 0.1;
timemax = 12;
% 投影回各自的参数范围

preXvel_sample = (preXvelmax - preXvelmin).*X(:,1) + preXvelmin;
preAx_sample = (preAxmax - preAxmin).*X(:,2) + preAxmin;
time_sample = (timemax - timemin).*X(:,3) + timemin;
% 组合成测试矩阵
LatinCubeSim = cat(2,preXvel_sample,preAx_sample,time_sample);
%% 3维绘图
plot3(LatinCubeSim(:,1),LatinCubeSim(:,2),LatinCubeSim(:,3),'k.')
save  Lh200.txt  LatinCubeSim -ascii 
