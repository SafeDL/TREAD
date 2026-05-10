%% 多元概率密度估计,效果不好
clc,clear all
% 加载已经保存过的数据，提取highD中0-150m的有效跟随数据，共10维特征
load('Longdata.mat'); % 全部有效跟车数据
%% 定义每个维度的采样区间
data = [dhw(1:100),xVelocity(1:100),precedingXVelocity(1:100)];
griddhw = linspace(min(dhw(1:100)),max(dhw(1:100)),100);
gridxVelocity = linspace(min(xVelocity(1:100)),max(xVelocity(1:100)),100);
gridprecedingXVelocity = linspace(min(precedingXVelocity(1:100)),max(precedingXVelocity(1:100)),100);
gridrVelocity = linspace(-25,25,1000);
gridxAcceleration = linspace(-7,7,1000);
gridTHW = linspace(0,20,1000);
gridTTC = linspace(0,19.5,1000);
gridrpreAx = linspace(-7,7,1000);
griddectime = linspace(0.5,42,1000);
gridaccetime = linspace(0.5,47,1000);
% 整理采样点
% [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10] = ndgrid(griddhw,gridxVelocity,gridprecedingXVelocity,gridrVelocity,gridxAcceleration,gridTHW,gridTTC,...
%     gridrpreAx,griddectime,gridaccetime);
% x1 = x1(:,:)';
% x2 = x2(:,:)';
% x3 = x3(:,:)';
% x4 = x4(:,:)';
% x5 = x5(:,:)';
% x6 = x6(:,:)';
% x7 = x7(:,:)';
% x8 = x8(:,:)';
% x9 = x9(:,:)';
% x10 = x10(:,:)';
% xi = [x1(:) x2(:) x3(:) x4(:) x5(:) x6(:) x7(:) x8(:) x9(:) x10(:)];

% 先以三个维度为例
[x1,x2,x3] = ndgrid(griddhw,gridxVelocity,gridprecedingXVelocity);
x1 = x1(:,:)';
x2 = x2(:,:)';
x3 = x3(:,:)';
xi = [x1(:) x2(:) x3(:)];
%% 带宽的经验公式
d = 3; % 维度个数
n = length(dhw); % 观测值个数
b1 = std(dhw)*(4/((d+2)*n))^(1/(d+4))
b2 = std(xVelocity)*(4/((d+2)*n))^(1/(d+4))
b3 = std(precedingXVelocity)*(4/((d+2)*n))^(1/(d+4))
%% 多元变量核概率密度估计
tic
f = mvksdensity(data,xi,...
	'Bandwidth',[b1 b2 b3],...
	'Kernel','normpdf');
toc
%% 寻找最佳方差-MH采样
tic
sample = Multi_MH_Sampling(f,xi);
toc
%% 比较采样和原始结果统计分布
% 得到直方图分布
[f_ecdf_dhw_sample, xc_ecdf_dhw_sample] = ecdf(sample(:,1));
[f_ecdf_xVel_sample, xc_ecdf_xVel_sample] = ecdf(sample(:,2));
[f_ecdf_precedingVel_sample, xc_ecdf_precedingVel_sample] = ecdf(sample(:,3));
% [f_ecdf_rVel_sample, xc_ecdf_rVel_sample] = ecdf(sample(:,4));
% [f_ecdf_preAx_sample, xc_ecdf_preAx_sample] = ecdf(sample(:,5));
% [f_ecdf_dectime_sample, xc_ecdf_dectime_sample] = ecdf(sample(:,6));
% [f_ecdf_accetime_sample, xc_ecdf_accetime_sample] = ecdf(sample(:,7));
% 得到核概率密度估计
pts = linspace(min(sample(:,1)),max(sample(:,1)),100);
[f_ks_dhw_sample,xi_dhw_sample] = ksdensity(sample(:,1),pts);% u1为窗宽

pts = linspace(min(sample(:,2)),max(sample(:,2)),100);
[f_ks_xVelocity_sample,xi_xVelocity_sample] = ksdensity(sample(:,2),pts);% u1为窗宽

pts = linspace(min(sample(:,3)),max(sample(:,3)),100);
[f_ks_precedingXVelocity_sample,xi_precedingXVelocity_sample] = ksdensity(sample(:,3),pts);% u1为窗宽
%% 绘制直方图和采样后概率密度
figure(1);
ecdfhist(f_ecdf_dhw_sample, xc_ecdf_dhw_sample,100);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
plot(xi_dhw_sample,f_ks_dhw_sample,'red','linewidth',3)
title('MH采样后跟车距离分布')
grid on;

figure(2);
ecdfhist(f_ecdf_xVel_sample, xc_ecdf_xVel_sample,100);
hold on;
xlabel('自车速度 (m/s)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xVelocity_sample,f_ks_xVelocity_sample,'red','linewidth',3)
title('MH采样后自车速度分布')
grid on;

% 前车速度，m/s
figure(3);
ecdfhist(f_ecdf_precedingVel_sample, xc_ecdf_precedingVel_sample,100);
hold on;
xlabel('前车速度 (m/s)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_precedingXVelocity_sample,f_ks_precedingXVelocity_sample,'red','linewidth',3)
grid on;
title('MH采样后前车速度分布'); 
%% Overlay the theoretical density，将原始直方图、原始概率密度估计和MH采样的概率密度估计叠绘
[f_ecdf_dhw, xc_ecdf_dhw] = ecdf(data(:,1));
[f_ecdf_xVelocity, xc_ecdf_xVelocity] = ecdf(data(:,2));
[f_ecdf_precedingXVelocity, xc_ecdf_precedingXVelocity] = ecdf(data(:,3));

pts = linspace(min(dhw(1:1000)),max(dhw(1:1000)),100);
[f_ks_dhw,xi_dhw] = ksdensity(data(:,1),pts);

pts = linspace(min(xVelocity(1:1000)),max(xVelocity(1:1000)),100);
[f_ks_xVelocity,xi_xVelocity] = ksdensity(data(:,2),pts);

pts = linspace(min(precedingXVelocity(1:1000)),max(precedingXVelocity(1:1000)),100);
[f_ks_precedingXVelocity,xi_precedingXVelocity] = ksdensity(data(:,3),pts);


% 自车距离，m
figure(1);
ecdfhist(f_ecdf_dhw, xc_ecdf_dhw,100);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dhw,f_ks_dhw,'red','linewidth',3)
hold on;
plot(xi_dhw_sample,f_ks_dhw_sample,'blue','linewidth',3)
title('距离分布对比')
grid on;

% 自车速度，m/s
figure(2);
ecdfhist(f_ecdf_xVelocity, xc_ecdf_xVelocity,100);
hold on;
xlabel('自车速度 (m/s)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xVelocity,f_ks_xVelocity,'red','linewidth',3)
hold on;
plot(xi_xVelocity_sample,f_ks_xVelocity_sample,'blue','linewidth',3)
title('速度分布对比')
grid on;

% 前车速度，m/s
figure(3);
ecdfhist(f_ecdf_precedingXVelocity, xc_ecdf_precedingXVelocity,100);
hold on;
xlabel('前车速度 (m/s)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_precedingXVelocity,f_ks_precedingXVelocity,'red','linewidth',3)
hold on;
plot(xi_precedingXVelocity_sample,f_ks_precedingXVelocity_sample,'blue','linewidth',3)
grid on;
title('速度分布对比'); 

%% 创建边缘为核概率密度估计的散点图
figure('color','w');
tbl = table(dhw_sample,xVel_sample,rVel_sample,preAx_sample,accetime_sample,dectime_sample);
% 边缘为直方图
% s = scatterhistogram(tbl,'dhw_sample','xVel_sample', ...
%     'HistogramDisplayStyle','bar','NumBins',50,'LineWidth',1.2);

% 边缘为概率密度
s = scatterhistogram(tbl,'dhw_sample','xVel_sample', ...
    'HistogramDisplayStyle','smooth','LineWidth',1.2);
s.XLabel = 'DHW';
s.YLabel = 'xVel';