%% 提取highD的跟随工况参数，将60组数据归纳整合
tic
clc,clear all;
data = []; %变量初始化，highD中所有的跟随工况
for i = 1:9
    disp('i=')
    disp(i)
    videoString = "0"+ num2str(i);
    tracksFilename = sprintf('data/%s_tracks.csv', videoString);
    tracksStaticFilename = sprintf('data/%s_tracksMeta.csv', videoString);
    tracks = readInTracksCsv(tracksFilename, tracksStaticFilename);
    [tracks_filter,tracks_long,Result] = longfilter_onlycar(tracks);
    data = [data;Result];
end

for i = 10:60 
    disp('i=') 
    disp(i)
    videoString = num2str(i);
    tracksFilename = sprintf('data/%s_tracks.csv', videoString);
    tracksStaticFilename = sprintf('data/%s_tracksMeta.csv', videoString);
    tracs = readInTracksCsv(tracksFilename, tracksStaticFilename);
    [tracks_filter,tracks_long,Result] = longfilter_onlycar(tracks);
    data = [data;Result];
end
toc
%% 提取感兴趣的特征
dhw = [];
xVelocity = [];
precedingXVelocity = [];
rVelocity = [];
xAcceleration = [];
THW = [];
TTC = [];
preAx = [];
dectime = [];
accetime = [];

result_dhw = {};
result_xVelocity = {};
result_precedingXVelocity = {};
result_rVelocity = {};
result_xAcceleration = {};
result_THW = {};
result_TTC = {};
result_preAx = {};
result_dectime = {};
result_accetime = {};

parfor i=1:length(data)
    result_dhw(i).dhw = data(i).dhw;
    result_xVelocity(i).xVelocity = data(i).xVelocity;
    result_precedingXVelocity(i).precedingXVelocity = data(i).precedingXVelocity;
    result_rVelocity(i).rVelocity = data(i).rVelocity;
    result_xAcceleration(i).xAcceleration = data(i).xAcceleration;
    result_THW(i).THW = data(i).THW;
    result_TTC(i).TTC = data(i).TTC;
	result_preAx(i).preAx = data(i).preAx;
    result_dectime(i).dectime = data(i).dectime;
	result_accetime(i).accetime = data(i).accetime;
end
test_dhw = struct2cell(result_dhw);
test_xVelocity = struct2cell(result_xVelocity);
test_precedingXVelocity = struct2cell(result_precedingXVelocity);
test_rVelocity = struct2cell(result_rVelocity);
test_xAcceleration = struct2cell(result_xAcceleration);
test_THW = struct2cell(result_THW);
test_TTC = struct2cell(result_TTC);
test_preAx = struct2cell(result_preAx);
test_dectime = struct2cell(result_dectime);
test_accetime = struct2cell(result_accetime);

tic
parfor i = 1:length(data)
    dhw = [dhw;cell2mat(test_dhw(i))];
    xVelocity = [xVelocity;cell2mat(test_xVelocity(i))];
	precedingXVelocity = [precedingXVelocity;abs(cell2mat(test_precedingXVelocity(i)))];
	rVelocity = [rVelocity;cell2mat(test_rVelocity(i))];
	xAcceleration = [xAcceleration;cell2mat(test_xAcceleration(i))];
	THW = [THW;cell2mat(test_THW(i))];
	TTC = [TTC;cell2mat(test_TTC(i))];
	preAx = [preAx;cell2mat(test_preAx(i))];
	dectime = [dectime;cell2mat(test_dectime(i))];
	accetime = [accetime;cell2mat(test_accetime(i))];
end
toc
% 删除无效TTC，说明前车在加速
index = find((TTC == -1)|(TTC > 20));
TTC(index) = [];
% 删除异常THW
index = find(THW > 20);
THW(index) = [];
%% 程序入口，加载已经保存过的数据，提取highD中0-150m的有效跟随数据，共10维特征
load('Longdata.mat'); % 全部有效跟车数据
% 调用ecdf函数计算xc处的经验分布函数值f_ecd
[f_ecdf_dhw, xc_ecdf_dhw] = ecdf(dhw);
[f_ecdf_xVelocity, xc_ecdf_xVelocity] = ecdf(xVelocity);
[f_ecdf_precedingXVelocity, xc_ecdf_precedingXVelocity] = ecdf(precedingXVelocity);
[f_ecdf_rVelocity, xc_ecdf_rVelocity] = ecdf(rVelocity);
[f_ecdf_xAcceleration, xc_ecdf_xAcceleration] = ecdf(xAcceleration);
[f_ecdf_THW, xc_ecdf_THW] = ecdf(THW);
[f_ecdf_TTC, xc_ecdf_TTC] = ecdf(TTC);
[f_ecdf_preAx, xc_ecdf_preAx] = ecdf(preAx);
[f_ecdf_dectime, xc_ecdf_dectime] = ecdf(dectime);
[f_ecdf_accetime, xc_ecdf_accetime] = ecdf(accetime);
%% 绘制感兴趣的频率直方图
figure(1);
ecdfhist(f_ecdf_dhw, xc_ecdf_dhw,100);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(0,150,1000);
[f_ks_dhw,xi_dhw,u_dhw] = ksdensity(dhw,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dhw,f_ks_dhw,'red','linewidth',3)
title('跟车距离分布')
grid on;

% 自车速度，m/s
figure(2);
ecdfhist(f_ecdf_xVelocity, xc_ecdf_xVelocity,100);
hold on;
xlabel('自车速度 (m/s)');  % 为X轴加标签
xlim([0 50]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(0,50,1000);
[f_ks_xVelocity,xi_xVelocity,u_xVelocity] = ksdensity(xVelocity,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xVelocity,f_ks_xVelocity,'red','linewidth',3)
title('自车速度分布')
grid on;

% 前车速度，m/s
figure(3);
ecdfhist(f_ecdf_precedingXVelocity, xc_ecdf_precedingXVelocity,100);
hold on;
xlabel('前车速度 (m/s)');  % 为X轴加标签
xlim([0 50]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(0,50,1000);
[f_ks_precedingXVelocity,xi_precedingXVelocity,u_precedingXVelocity] = ksdensity(precedingXVelocity,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_precedingXVelocity,f_ks_precedingXVelocity,'red','linewidth',3)
grid on;
title('前车速度分布'); 

% 相对速度，m/s
figure(4);
ecdfhist(f_ecdf_rVelocity, xc_ecdf_rVelocity,100);
hold on;
xlabel('相对速度 (m/s)');  % 为X轴加标签
xlim([-25 25]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-25,25,1000);
[f_ks_rVelocity,xi_rVelocity,u_rVelocity] = ksdensity(rVelocity,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_rVelocity,f_ks_rVelocity,'red','linewidth',3)
grid on;
title('相对速度分布')

% 自车加速度，m/s2
figure(5);
ecdfhist(f_ecdf_xAcceleration, xc_ecdf_xAcceleration,100);
hold on;
xlabel('自车加速度 (m/s2)');  % 为X轴加标签
xlim([-7 7]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-7,7,1000);
[f_ks_xAcceleration,xi_xAcceleration,u_xAcceleration] = ksdensity(xAcceleration,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xAcceleration,f_ks_xAcceleration,'red','linewidth',3)
grid on;
title('自车加速度分布')

% 跟车时距分布，s
figure(6);
ecdfhist(f_ecdf_THW, xc_ecdf_THW,100);
hold on;
xlabel('时距 (s)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(0,20,1000);
[f_ks_THW,xi_THW,u_THW] = ksdensity(THW,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_THW,f_ks_THW,'red','linewidth',3)
grid on;
title('跟车时距分布')

% 碰撞时间分布，s
figure(7);
ecdfhist(f_ecdf_TTC, xc_ecdf_TTC,100);
hold on;
xlabel('时间 (s)');  % 为X轴加标签
xlim([0 19.5]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(0,19.5,1000);
% [f_ks_TTC,xi_TTC,u_TTC] = ksdensity(TTC,pts,'Support','positive','BoundaryCorrection','reflection');% u1为窗宽
[f_ks_TTC,xi_TTC,u_TTC] = ksdensity(TTC,pts);% u1为窗宽

% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_TTC,f_ks_TTC,'red','linewidth',3)
grid on;
title('碰撞时间分布')
 
% 前车加速度分布，m/s^2
figure(8);
ecdfhist(f_ecdf_preAx, xc_ecdf_preAx,100);
hold on;
xlabel('加速度 (m/s2)');  % 为X轴加标签
xlim([-7 7]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-7,7,1000);
[f_ks_preAx,xi_preAx,u_preAx] = ksdensity(preAx,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_preAx,f_ks_preAx,'red','linewidth',3)
grid on;
title('前车加速度分布');

% 前车减速时间分布，s
figure(9);
ecdfhist(f_ecdf_dectime, xc_ecdf_dectime,100);
hold on;
xlabel('减速时间 (s)');  % 为X轴加标签
xlim([0.5 42]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(0.5,42,1000);
[f_ks_dectime,xi_dectime,u_dectime] = ksdensity(dectime,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dectime,f_ks_dectime,'red','linewidth',3)
grid on;
title('前车减速时间分布')

% 前车加速时间分布，s
figure(10);
ecdfhist(f_ecdf_accetime, xc_ecdf_accetime,100);
hold on;
xlabel('加速时间 (s)');  % 为X轴加标签
xlim([0.5 47]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(0.5,47,1000);
[f_ks_accetime,xi_accetime,u_accetime] = ksdensity(accetime,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_accetime,f_ks_accetime,'red','linewidth',3)
grid on;
title('前车加速时间分布')
%% 寻找最佳方差,计算KL散度并返回最优值
tic
clc
T = 200; % Set the maximum number of iterations
dhw_sample_dataset = zeros(T,3001);
xVel_sample_dataset = zeros(T,3001);
precedingVel_sample_dataset = zeros(T,3001);
rVel_sample_dataset = zeros(T,3001);
preAx_sample_dataset = zeros(T,3001);
dectime_sample_dataset = zeros(T,3001);
accetime_sample_dataset = zeros(T,3001);
ii = 1;
for i = 0:0.01:30 
    dhw_sample_dataset(:,ii) = MH_Sampling(f_ks_dhw,i,xi_dhw,0,150);
    xVel_sample_dataset(:,ii)= MH_Sampling(f_ks_xVelocity,i,xi_xVelocity,0,50);
    precedingVel_sample_dataset(:,ii)= MH_Sampling(f_ks_precedingXVelocity,i,xi_precedingXVelocity,0,50);
	rVel_sample_dataset(:,ii) = MH_Sampling(f_ks_rVelocity,i,xi_rVelocity,-25,25);
    preAx_sample_dataset(:,ii) = MH_Sampling(f_ks_preAx,i,xi_preAx,-7,6);  
    dectime_sample_dataset(:,ii) = MH_Sampling(f_ks_dectime,i,xi_dectime,0,50);
	accetime_sample_dataset(:,ii) = MH_Sampling(f_ks_accetime,i,xi_accetime,0,50);
    ii = ii + 1; 
end
toc

dhw_sample_KL = zeros(1,3001);
xVel_sample_KL = zeros(1,3001);
precedingVel_sample_KL = zeros(1,3001);
rVel_sample_KL = zeros(1,3001);
preAx_sample_KL = zeros(1,3001);
dectime_sample_KL = zeros(1,3001);  
accetime_sample_KL = zeros(1,3001);

for i = 1:3001
    pts = linspace(0,150,1000);
    [f_ks,xi,u] = ksdensity(dhw_sample_dataset(:,i),pts);
    dhw_sample_KL(i) = abs(KL_Calculate(f_ks_dhw,f_ks));
end

for i = 1:3001
    pts = linspace(0,50,1000);
    [f_ks,xi,u] = ksdensity(xVel_sample_dataset(:,i),pts);
    xVel_sample_KL(i) = abs(KL_Calculate(f_ks_xVelocity,f_ks));
end

for i = 1:3001
    pts = linspace(0,50,1000);
    [f_ks,xi,u] = ksdensity(precedingVel_sample_dataset(:,i),pts);
    precedingVel_sample_KL(i) = abs(KL_Calculate(f_ks_precedingXVelocity,f_ks));
end

for i = 1:3001
    pts = linspace(-25,25,1000);
    [f_ks,xi,u] = ksdensity(rVel_sample_dataset(:,i),pts);
    rVel_sample_KL(i) = abs(KL_Calculate(f_ks_rVelocity,f_ks));
end

for i = 1:3001
    pts = linspace(-7,7,1000);
    [f_ks,xi,u] = ksdensity(preAx_sample_dataset(:,i),pts);
    preAx_sample_KL(i) = abs(KL_Calculate(f_ks_preAx,f_ks));
end

for i = 1:3001
    pts = linspace(0.5,42,1000);
    [f_ks,xi,u] = ksdensity(dectime_sample_dataset(:,i),pts);
    dectime_sample_KL(i) = abs(KL_Calculate(f_ks_dectime,f_ks));
end

for i = 1:3001
    pts = linspace(0.5,47,1000);
    [f_ks,xi,u] = ksdensity(accetime_sample_dataset(:,i),pts);
    accetime_sample_KL(i) = abs(KL_Calculate(f_ks_accetime,f_ks));
end

i = 0:0.01:30;
% 输出最优解
index_dhw = find(dhw_sample_KL == min(dhw_sample_KL));
disp('跟车距离最小KL散度值为：')
dhw_sample_KL(index_dhw)
disp('对应的采样标准差：')
i(index_dhw)

index_xVel = find(xVel_sample_KL == min(xVel_sample_KL));
disp('自车速度的最小KL散度值为：')
xVel_sample_KL(index_xVel)
disp('对应的采样标准差：')
i(index_xVel)

index_precedingVel = find(precedingVel_sample_KL == min(precedingVel_sample_KL));
disp('前车速度的最小KL散度值为：')
precedingVel_sample_KL(index_precedingVel)
disp('对应的采样标准差：')
i(index_precedingVel)

index_rVel = find(rVel_sample_KL == min(rVel_sample_KL));
disp('相对速度的最小KL散度值为：')
rVel_sample_KL(index_rVel)
disp('对应的采样标准差：')
i(index_rVel)

index_preAx = find(preAx_sample_KL == min(preAx_sample_KL));
disp('前车加速度的最小KL散度值为：')
preAx_sample_KL(index_preAx)
disp('对应的采样标准差：')
i(index_preAx)

index_dectime = find(dectime_sample_KL == min(dectime_sample_KL));
disp('前车减速时间的最小KL散度值为：')
dectime_sample_KL(index_dectime)
disp('对应的采样标准差：')
i(index_dectime)

index_accetime = find(accetime_sample_KL == min(accetime_sample_KL));
disp('前车加速时间的最小KL散度值为：')
accetime_sample_KL(index_accetime)
disp('对应的采样标准差：')
i(index_accetime)
%% 根据上述最优解，提取MH采样结果
dhw_sample = dhw_sample_dataset(:,index_dhw);
xVel_sample = xVel_sample_dataset(:,index_xVel);
precedingVel_sample = precedingVel_sample_dataset(:,index_precedingVel);
rVel_sample = rVel_sample_dataset(:,index_rVel);
preAx_sample = preAx_sample_dataset(:,index_preAx);
dectime_sample = dectime_sample_dataset(:,index_dectime);
accetime_sample = accetime_sample_dataset(:,index_accetime);
%% 使用Slice sampling，不需要设计建议概率密度
load('fittedmodel.mat')
clc
tic
dhw_sample= Slice_Sampling(fittedmodel_dhw,0,150);
xVel_sample= Slice_Sampling(fittedmodel_xVel,0,50);
precedingVel_sample = Slice_Sampling(fittedmodel_preVel,0,50);
rVel_sample = Slice_Sampling(fittedmodel_rVel,-25,25);
preAx_sample = Slice_Sampling(fittedmodel_preAx,-7,6);
dectime_sample = Slice_Sampling(fittedmodel_dectime,0,50);
accetime_sample = Slice_Sampling(fittedmodel_accetime,0,50);
toc
%% 绘制直方图和采样后概率密度
% 得到直方图分布
[f_ecdf_dhw_sample, xc_ecdf_dhw_sample] = ecdf(dhw_sample);
[f_ecdf_xVel_sample, xc_ecdf_xVel_sample] = ecdf(xVel_sample);
[f_ecdf_precedingVel_sample, xc_ecdf_precedingVel_sample] = ecdf(precedingVel_sample);
[f_ecdf_rVel_sample, xc_ecdf_rVel_sample] = ecdf(rVel_sample);
[f_ecdf_preAx_sample, xc_ecdf_preAx_sample] = ecdf(preAx_sample);
[f_ecdf_dectime_sample, xc_ecdf_dectime_sample] = ecdf(dectime_sample);
[f_ecdf_accetime_sample, xc_ecdf_accetime_sample] = ecdf(accetime_sample);
% 得到核概率密度估计
pts = linspace(0,150,1000);
[f_ks_dhw_sample,xi_dhw_sample] = ksdensity(dhw_sample,pts);% u1为窗宽
pts = linspace(0,50,1000);
[f_ks_xVelocity_sample,xi_xVelocity_sample] = ksdensity(xVel_sample,pts);% u1为窗宽
pts = linspace(0,50,1000);
[f_ks_precedingXVelocity_sample,xi_precedingXVelocity_sample] = ksdensity(precedingVel_sample,pts);% u1为窗宽
pts = linspace(-25,25,1000);
[f_ks_rVelocity_sample,xi_rVelocity_sample] = ksdensity(rVel_sample,pts);% u1为窗宽
pts = linspace(-7,7,1000);
[f_ks_preAx_sample,xi_preAx_sample] = ksdensity(preAx_sample,pts);% u1为窗宽
pts = linspace(0.5,42,1000);
[f_ks_dectime_sample,xi_dectime_sample] = ksdensity(dectime_sample,pts);% u1为窗宽
pts = linspace(0.5,47,1000);
[f_ks_accetime_sample,xi_accetime_sample] = ksdensity(accetime_sample,pts);% u1为窗宽

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
xlim([0 50]);
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
xlim([0 50]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_precedingXVelocity_sample,f_ks_precedingXVelocity_sample,'red','linewidth',3)
grid on;
title('MH采样后前车速度分布'); 

% 相对速度，m/s
figure(4);
ecdfhist(f_ecdf_rVel_sample, xc_ecdf_rVel_sample,100);
hold on;
xlabel('相对速度 (m/s)');  % 为X轴加标签
xlim([-25 25]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_rVelocity_sample,f_ks_rVelocity_sample,'red','linewidth',3)
grid on;
title('MH采样后相对速度分布')

% 前车加速度分布，m/s^2
figure(5);
ecdfhist(f_ecdf_preAx_sample, xc_ecdf_preAx_sample,100);
hold on;
xlabel('加速度 (m/s2)');  % 为X轴加标签
xlim([-7 7]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_preAx_sample,f_ks_preAx_sample,'red','linewidth',3)
grid on;
title('MH采样后前车加速度分布');

% 前车减速时间分布，s
figure(6);
ecdfhist(f_ecdf_dectime_sample, xc_ecdf_dectime_sample,100);
hold on;
xlabel('减速时间 (s)');  % 为X轴加标签
xlim([0.5 42]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dectime_sample,f_ks_dectime_sample,'red','linewidth',3)
grid on;
title('MH采样后前车减速时间分布')

% 前车加速时间分布，s
figure(7);
ecdfhist(f_ecdf_accetime_sample, xc_ecdf_accetime_sample,100);
hold on;
xlabel('加速时间 (s)');  % 为X轴加标签
xlim([0.5 47]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_accetime_sample,f_ks_accetime_sample,'red','linewidth',3)
grid on;
title('MH采样后前车加速时间分布')
%% 将原始直方图、原始概率密度估计和MH采样的概率密度估计叠绘
% 跟车距离处理,m
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
xlim([0 50]);
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
xlim([0 50]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_precedingXVelocity,f_ks_precedingXVelocity,'red','linewidth',3)
hold on;
plot(xi_precedingXVelocity_sample,f_ks_precedingXVelocity_sample,'blue','linewidth',3)
grid on;
title('速度分布对比'); 

% 相对速度，m/s
figure(4);
ecdfhist(f_ecdf_rVelocity, xc_ecdf_rVelocity,100);
hold on;
xlabel('相对速度 (m/s)');  % 为X轴加标签
xlim([-25 25]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_rVelocity,f_ks_rVelocity,'red','linewidth',3)
hold on;
plot(xi_rVelocity_sample,f_ks_rVelocity_sample,'blue','linewidth',3)
grid on;
title('速度分布对比')

% 前车加速度分布，m/s^2
figure(5);
ecdfhist(f_ecdf_preAx, xc_ecdf_preAx,100);
hold on;
xlabel('加速度 (m/s2)');  % 为X轴加标签
xlim([-7 7]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_preAx,f_ks_preAx,'red','linewidth',3)
hold on;
plot(xi_preAx_sample,f_ks_preAx_sample,'blue','linewidth',3)
grid on;
title('加速度分布对比');

% 前车减速时间分布，s
figure(6);
ecdfhist(f_ecdf_dectime, xc_ecdf_dectime,100);
hold on;
xlabel('减速时间 (s)');  % 为X轴加标签
xlim([0.5 42]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dectime,f_ks_dectime,'red','linewidth',3)
hold on;
plot(xi_dectime_sample,f_ks_dectime_sample,'blue','linewidth',3)
grid on;
title('减速时间分布对比')

% 前车加速时间分布，s
figure(7);
ecdfhist(f_ecdf_accetime, xc_ecdf_accetime,100);
hold on;
xlabel('加速时间 (s)');  % 为X轴加标签
xlim([0.5 47]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_accetime,f_ks_accetime,'red','linewidth',3)
hold on;
plot(xi_accetime_sample,f_ks_accetime_sample,'blue','linewidth',3)
grid on;
title('加速时间分布对比')
%% 堆叠成测试数据
sample_200 = zeros(200,6);
sample_200(:,1) = dhw_sample;
sample_200(:,2) = xVel_sample;
sample_200(:,3) = rVel_sample;
sample_200(:,4) = preAx_sample;
sample_200(:,5) = accetime_sample;
sample_200(:,6) = dectime_sample;
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