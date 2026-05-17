%% 提取highD的超车工况参数 将60组数据归纳整合
clc,clear all;
data = []; %变量初始化，highD中所有的跟随工况
for i = 1:9
    disp(i);
    videoString = "0"+ num2str(i);
    tracksFilename = sprintf('data/%s_tracks.csv', videoString);
    tracksStaticFilename = sprintf('data/%s_tracksMeta.csv', videoString);
    tracks = readInTracksCsv(tracksFilename, tracksStaticFilename);
    [overtakefilter,Result] = OvertakeFilter(tracks);
    data = [data;Result];
end

for i = 10:60
	disp(i);
    videoString = num2str(i);
    tracksFilename = sprintf('data/%s_tracks.csv', videoString);
    tracksStaticFilename = sprintf('data/%s_tracksMeta.csv', videoString);
    tracks = readInTracksCsv(tracksFilename, tracksStaticFilename);
    [overtakefilter,Result] = OvertakeFilter(tracks);
    data = [data;Result];
end
%% 提取感兴趣的特征
xVelocity = [];
yVelocity = [];
xAcceleration = [];
yAcceleration = [];
duration = [];
dx = [];
InitDhw = [];
InitFollowingXVelocity = [];
FollowingXVelocity = [];
InitFollowingXAcce = [];
FollowingXAcce = [];

tic
parfor i=1:length(data)
    temp_xVelocity = cell2mat(data(i)).xVelocity;
	temp_yVelocity = cell2mat(data(i)).yVelocity;
	temp_xAcceleration = cell2mat(data(i)).xAcceleration;
	temp_yAcceleration = cell2mat(data(i)).yAcceleration;
	temp_duration = cell2mat(data(i)).duration;
    temp_dx = cell2mat(data(i)).dx;
	temp_InitDhw = cell2mat(data(i)).InitDhw;
	temp_InitFollowingXVelocity = cell2mat(data(i)).InitFollowingXVelocity;
	temp_FollowingXVelocity = cell2mat(data(i)).FollowingXVelocity;
	temp_InitFollowingXAcce = cell2mat(data(i)).InitFollowingXAcce;
	temp_FollowingXAcce = cell2mat(data(i)).FollowingXAcce;
    xVelocity = [xVelocity;temp_xVelocity];
	yVelocity = [yVelocity;temp_yVelocity];
    xAcceleration = [xAcceleration;temp_xAcceleration];
    yAcceleration = [yAcceleration;temp_yAcceleration];
    duration = [duration;temp_duration];
    dx = [dx;temp_dx];
    InitDhw = [InitDhw;temp_InitDhw];
    InitFollowingXVelocity = [InitFollowingXVelocity;temp_InitFollowingXVelocity];
    FollowingXVelocity = [FollowingXVelocity;temp_FollowingXVelocity];
	InitFollowingXAcce = [InitFollowingXAcce;temp_InitFollowingXAcce];
    FollowingXAcce = [FollowingXAcce;temp_FollowingXAcce];
end
toc
%% 绘制感兴趣的频率直方图，程序入口
% 加载已经保存过的数据，提取highD中有效切入数据，共10维特征
% load('Latdata.mat'); % 全部有效超车数据
% 调用ecdf函数计算xc处的经验分布函数值f_ecd
[f_ecdf_xVelocity, xc_ecdf_xVelocity] = ecdf(xVelocity);
[f_ecdf_yVelocity, xc_ecdf_yVelocity] = ecdf(yVelocity);
[f_ecdf_xAcceleration, xc_ecdf_xAcceleration] = ecdf(xAcceleration);
[f_ecdf_yAcceleration, xc_ecdf_yAcceleration] = ecdf(yAcceleration);
[f_ecdf_duration, xc_ecdf_duration] = ecdf(duration);
[f_ecdf_dx, xc_ecdf_dx] = ecdf(dx);
[f_ecdf_InitDhw, xc_ecdf_InitDhw] = ecdf(InitDhw);
[f_ecdf_InitFollowingXVelocity, xc_ecdf_InitFollowingXVelocity] = ecdf(InitFollowingXVelocity);
[f_ecdf_FollowingXVelocity, xc_ecdf_FollowingXVelocity] = ecdf(FollowingXVelocity);
[f_ecdf_InitFollowingXAcce, xc_ecdf_InitFollowingXAcce] = ecdf(InitFollowingXAcce);
[f_ecdf_FollowingXAcce, xc_ecdf_FollowingXAcce] = ecdf(FollowingXAcce);
%% 频率直方图绘图
clc;
figure(1);
ecdfhist(f_ecdf_xVelocity, xc_ecdf_xVelocity,100);
hold on;
xlabel('纵向速度 (m/s)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(2,55,1000);
[f_ks_xVelocity,xi_xVelocity] = ksdensity(xVelocity,pts);
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xVelocity,f_ks_xVelocity,'red','linewidth',3)
title('超车过程纵向速度分布')
grid on;

% 超车过程横向速度分布
figure(2);
ecdfhist(f_ecdf_yVelocity, xc_ecdf_yVelocity,100);
hold on;
xlabel('横向速度 (m/s)');  % 为X轴加标签
xlim([-2.5 2.5]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-2.5,2.5,1000);
[f_ks_yVelocity,xi_yVelocity] = ksdensity(yVelocity,pts);
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_yVelocity,f_ks_yVelocity,'red','linewidth',3)
title('超车过程横向速度分布')
grid on;

% 超车过程纵向加速度分布
figure(3);
ecdfhist(f_ecdf_xAcceleration, xc_ecdf_xAcceleration,100);
hold on;
xlabel('纵向加速度 (m/s^2)');  % 为X轴加标签
xlim([-5 4]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-5,4,1000);
[f_ks_xAcceleration,xi_xAcceleration] = ksdensity(xAcceleration,pts);
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xAcceleration,f_ks_xAcceleration,'red','linewidth',3)
grid on;
title('超车过程纵向加速度分布'); 

% 超车过程横向加速度分布
figure(4);
ecdfhist(f_ecdf_yAcceleration, xc_ecdf_yAcceleration,100);
hold on;
xlabel('横向加速度 (m/s^2)');  % 为X轴加标签
xlim([-1.5 1.5]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-1.5,1.5,1000);
[f_ks_yAcceleration,xi_yAcceleration] = ksdensity(yAcceleration,pts);
plot(xi_yAcceleration,f_ks_yAcceleration,'red','linewidth',3)
grid on;
title('超车过程横向加速度分布')

% 超车时间分布
figure(5);
ecdfhist(f_ecdf_duration, xc_ecdf_duration,100);
hold on;
xlabel('时间 (s)');  % 为X轴加标签
xlim([1 16]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(1,16,1000);
[f_ks_duration,xi_duration] = ksdensity(duration,pts);
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_duration,f_ks_duration,'red','linewidth',3)
grid on;
title('超车时间分布')

% 超车初始纵向距离分布
figure(6);
ecdfhist(f_ecdf_InitDhw, xc_ecdf_InitDhw,1000);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
xlim([-78,-5]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-78,-5,1000);
[f_ks_InitDhw,xi_InitDhw] = ksdensity(InitDhw,pts);
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_InitDhw,f_ks_InitDhw,'red','linewidth',3)
grid on;
title('超车初始纵向距离分布')

% 超车初始纵向车速
figure(7);
ecdfhist(f_ecdf_InitFollowingXVelocity, xc_ecdf_InitFollowingXVelocity,100);
hold on;
xlabel('初始速度 (m/s)');  % 为X轴加标签
xlim([4 58]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(4,58,1000);
[f_ks_InitFollowingXVelocity,xi_InitFollowingXVelocity] = ksdensity(InitFollowingXVelocity,pts);
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_InitFollowingXVelocity,f_ks_InitFollowingXVelocity,'red','linewidth',3)
grid on;
title('超车初始纵向车速');

% 跟车纵向速度分布
figure(8);
ecdfhist(f_ecdf_FollowingXVelocity, xc_ecdf_FollowingXVelocity,100);
hold on;
xlabel('纵向速度 (m/s)');  % 为X轴加标签
xlim([4 58]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(4,58,1000);
[f_ks_FollowingXVelocity,xi_FollowingXVelocity] = ksdensity(FollowingXVelocity,pts,'Support','positive','BoundaryCorrection','reflection');% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_FollowingXVelocity,f_ks_FollowingXVelocity,'red','linewidth',3)
grid on;
title('跟车纵向速度分布')

% 跟车初始纵向加速度分布
figure(9);
ecdfhist(f_ecdf_InitFollowingXAcce, xc_ecdf_InitFollowingXAcce,100);
hold on;
xlabel('加速度 (m/s^2)');  % 为X轴加标签
xlim([-3 2]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-3,2,1000);
[f_ks_InitFollowingXAcce,xi_InitFollowingXAcce] = ksdensity(InitFollowingXAcce,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_InitFollowingXAcce,f_ks_InitFollowingXAcce,'red','linewidth',3)
grid on;
title('跟车初始纵向加速度分布')

% 跟车纵向加速度分布
figure(10);
ecdfhist(f_ecdf_FollowingXAcce, xc_ecdf_FollowingXAcce,100);
hold on;
xlabel('加速时间 (s)');  % 为X轴加标签
xlim([-3.7 2]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(-3.7,2,1000);
[f_ks_FollowingXAcce,xi_FollowingXAcce] = ksdensity(FollowingXAcce,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_FollowingXAcce,f_ks_FollowingXAcce,'red','linewidth',3)
grid on;
title('跟车纵向加速度分布')

% 超车纵向距离分布
figure(11);
ecdfhist(f_ecdf_dx, xc_ecdf_dx,100);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
xlim([1 300]);
ylabel('f(x)');  % 为Y轴加标签
% 调用ksdensity函数进行核密度估计,u1为窗宽
pts = linspace(1,300,1000);
[f_ks_dx,xi_dx] = ksdensity(dx,pts);% u1为窗宽
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dx,f_ks_dx,'red','linewidth',3)
grid on;
title('超车纵向距离分布')
%% 寻找最佳方差，Initialize the Metropolis sampler
clc
T = 500; % Set the maximum number of iterations
% 先单独尝试对每个变量生成MCMC
InitDhw_sample = zeros(T,1);
dx_sample = zeros(T,1);
xVelocity_sample = zeros(T,1);
xAcceleration_sample = zeros(T,1);
InitFollowingXVelocity_sample = zeros(T,1);
seed=1; 
rand( 'state' , seed ); % set the random seed
randn( 'state' , seed );
% generate start value
InitDhw_sample(1) = unifrnd( -77.11 , -5.02 );
dx_sample(1) = unifrnd( 104.05 , 298.48 );
xVelocity_sample(1) = unifrnd( 10.01 , 46.7 );
xAcceleration_sample(1) = unifrnd( -2.63 , 1.52 );
InitFollowingXVelocity_sample(1) = unifrnd( 17.43 , 50.23 );
%% Start sampling
% 定义不同的选择值下的存储结果
InitDhw_sample_dataset = zeros(T,4001);
dx_sample_dataset = zeros(T,4001);
xVelocity_sample_dataset = zeros(T,501);
xAcceleration_sample_dataset = zeros(T,41);
InitFollowingXVelocity_sample_dataset = zeros(T,601);

ii = 1;
for i = 0:0.01:40 
    t = 1; 
    while t < T % Iterate until we have T samples
        t = t + 1;    
        % Propose a new value for theta using a normal proposal density 
        InitDhw_star = normrnd( InitDhw_sample(t-1) ,i );
        dx_star = normrnd( dx_sample(t-1) ,i );
        % Calculate the acceptance ratio of InitDhw
        index_InitDhw_star = find(abs(InitDhw_star - xi_InitDhw) == min(abs(InitDhw_star  - xi_InitDhw)));
        index_InitDhw = find(abs(InitDhw_sample(t-1) - xi_InitDhw) == min(abs(InitDhw_sample(t-1) - xi_InitDhw)));
        alpha_InitDhw = min( [ 1 f_ks_InitDhw( index_InitDhw_star ) / f_ks_InitDhw( index_InitDhw ) ] );
        % Calculate the acceptance ratio of xVel
        index_dx_star = find(abs(dx_star - xi_dx) == min(abs(dx_star  - xi_dx)));
        index_dx = find(abs(dx_sample(t-1) - xi_dx) == min(abs(dx_sample(t-1) - xi_dx)));
        alpha_dx = min( [ 1 f_ks_dx( index_dx_star ) / f_ks_dx( index_dx ) ] );
        % Draw a uniform deviate from [ 0 1 ]
        u = rand;     
        % Do we accept this proposal InitDhw?
        if (u < alpha_InitDhw) && (InitDhw_star >= -77.11) && (InitDhw_star <= -5.02)
            InitDhw_sample(t) = InitDhw_star; % If so, proposal becomes new state
        else
            InitDhw_sample(t) = InitDhw_sample(t-1); % If not, copy old state
        end  
        % Do we accept this proposal dx?
        if (u < alpha_dx) && (dx_star >= 104.05) && (dx_star <= 298.48)
            dx_sample(t) = dx_star; % If so, proposal becomes new state
        else
            dx_sample(t) = dx_sample(t-1); % If not, copy old state
        end  
    end   
    InitDhw_sample_dataset(:,ii) = InitDhw_sample;
    dx_sample_dataset(:,ii)= dx_sample;
    ii = ii + 1;
end

ii = 1;
for i = 0:0.01:5 
    t = 1; 
    while t < T % Iterate until we have T samples
        t = t + 1; 
        % Propose a new value for theta using a normal proposal density 
        xVelocity_star = normrnd(xVelocity_sample(t-1),i);    
        % Calculate the acceptance ratio
        index_xVelocity_star = find(abs(xVelocity_star - xi_xVelocity) == min(abs(xVelocity_star  - xi_xVelocity)));
        index_xVelocity = find(abs(xVelocity_sample(t-1) - xi_xVelocity) == min(abs(xVelocity_sample(t-1) - xi_xVelocity)));
        alpha = min( [ 1 f_ks_xVelocity( index_xVelocity_star ) / f_ks_xVelocity( index_xVelocity ) ] );
        % Draw a uniform deviate from [ 0 1 ]
        u = rand;     
        % Do we accept this proposal?
        if (u < alpha) && (xVelocity_star <= 46.7) && (xVelocity_star >= 10.01)
            xVelocity_sample(t) = xVelocity_star; % If so, proposal becomes new state
        else
            xVelocity_sample(t) = xVelocity_sample(t-1); % If not, copy old state
        end
    end   
    xVelocity_sample_dataset(:,ii) = xVelocity_sample;
	ii = ii + 1;
end

ii = 1;
for i = 0:0.01:0.4 
    t = 1; 
    while t < T % Iterate until we have T samples
        t = t + 1; 
        % Propose a new value for theta using a normal proposal density 
        xAcceleration_star = normrnd(xAcceleration_sample(t-1),i);
        % Calculate the acceptance ratio
        index_xAcceleration_star = find(abs(xAcceleration_star - xi_xAcceleration) == min(abs(xAcceleration_star  - xi_xAcceleration)));
        index_xAcceleration = find(abs(xAcceleration_sample(t-1) - xi_xAcceleration) == min(abs(xAcceleration_sample(t-1) - xi_xAcceleration)));
        alpha = min( [ 1 f_ks_xAcceleration( index_xAcceleration_star ) / f_ks_xAcceleration( index_xAcceleration ) ] );
        % Draw a uniform deviate from [ 0 1 ]
        u = rand;     
        % Do we accept this proposal?
        if (u < alpha) && (xAcceleration_star >= -2.63) && (xAcceleration_star <= 1.52)
            xAcceleration_sample(t) = xAcceleration_star; % If so, proposal becomes new state
        else
            xAcceleration_sample(t) = xAcceleration_sample(t-1); % If not, copy old state
        end
    end   
    xAcceleration_sample_dataset(:,ii) = xAcceleration_sample;
    ii = ii + 1;
end
    
ii = 1;
for i = 0:0.01:6
    t = 1; 
    while t < T % Iterate until we have T samples
        t = t + 1; 
        % Propose a new value for theta using a normal proposal density 
        InitFollowingXVelocity_star = normrnd(InitFollowingXVelocity_sample(t-1),i);  
        % Calculate the acceptance ratio of dectime
        index_InitFollowingXVelocity_star = find(abs(InitFollowingXVelocity_star - xi_InitFollowingXVelocity) == min(abs(InitFollowingXVelocity_star  - xi_InitFollowingXVelocity)));
        index_InitFollowingXVelocity = find(abs(InitFollowingXVelocity_sample(t-1) - xi_InitFollowingXVelocity) == min(abs(InitFollowingXVelocity_sample(t-1) - xi_InitFollowingXVelocity)));
        alpha_InitFollowingXVelocity = min( [ 1 f_ks_InitFollowingXVelocity( index_InitFollowingXVelocity_star ) / f_ks_InitFollowingXVelocity( index_InitFollowingXVelocity ) ] );
        % Draw a uniform deviate from [ 0 1 ]
        u = rand;     
        % Do we accept this proposal of dectime?
        if (u < alpha_InitFollowingXVelocity) && (InitFollowingXVelocity_star >=17.43) && (InitFollowingXVelocity_star <= 50.23)
            InitFollowingXVelocity_sample(t) = InitFollowingXVelocity_star; % If so, proposal becomes new state
        else
            InitFollowingXVelocity_sample(t) = InitFollowingXVelocity_sample(t-1); % If not, copy old state
        end 
    end   
    InitFollowingXVelocity_sample_dataset(:,ii) = InitFollowingXVelocity_sample;
    ii = ii + 1;
end
%% 计算KL散度并返回最优值
InitDhw_sample_KL = zeros(1,4001);
dx_sample_KL = zeros(1,4001);
xVelocity_sample_KL = zeros(1,501);
xAcceleration_sample_KL = zeros(1,41);
InitFollowingXVelocity_sample_KL = zeros(1,601);

for i = 1:4001
    pts = linspace(-78,-5,1000);
    [f_ks,xi,u] = ksdensity(InitDhw_sample_dataset(:,i),pts);
    InitDhw_sample_KL(i) = abs(KL_Calculate(f_ks_InitDhw,f_ks));
end

for i = 1:4001
    pts = linspace(1,300,1000);
    [f_ks,xi,u] = ksdensity(dx_sample_dataset(:,i),pts);
    dx_sample_KL(i) = abs(KL_Calculate(f_ks_dx,f_ks));
end

for i = 1:501
    pts = linspace(2,55,1000);
    [f_ks,xi,u] = ksdensity(xVelocity_sample_dataset(:,i),pts);
    xVelocity_sample_KL(i) = abs(KL_Calculate(f_ks_xVelocity,f_ks));
end

for i = 1:41
    pts = linspace(-5,4,1000);
    [f_ks,xi,u] = ksdensity(xAcceleration_sample_dataset(:,i),pts);
    xAcceleration_sample_KL(i) = abs(KL_Calculate(f_ks_xAcceleration,f_ks));
end

for i = 1:601
    pts = linspace(4,58,1000);
    [f_ks,xi,u] = ksdensity(InitFollowingXVelocity_sample_dataset(:,i),pts);
    InitFollowingXVelocity_sample_KL(i) = abs(KL_Calculate(f_ks_InitFollowingXVelocity,f_ks));
end

% 输出最优解
index_InitDhw = find(InitDhw_sample_KL == min(InitDhw_sample_KL));
disp('初始跟车距离最小KL散度值为：')
InitDhw_sample_KL(index_InitDhw)
disp('对应的采样标准差：')
i = 0:0.01:40;
i(index_InitDhw)

index_dx = find(dx_sample_KL == min(dx_sample_KL));
disp('超车行驶长度的最小KL散度值为：')
dx_sample_KL(index_dx)
disp('对应的采样标准差：')
i(index_dx)

index_xVelocity = find(xVelocity_sample_KL == min(xVelocity_sample_KL));
disp('超车纵向行驶速度的最小KL散度值为：')
xVelocity_sample_KL(index_xVelocity)
disp('对应的采样标准差：')
i = 0:0.01:5;
i(index_xVelocity)

index_xAcceleration = find(xAcceleration_sample_KL == min(xAcceleration_sample_KL));
disp('超车行驶纵向加速度的最小KL散度值为：')
xAcceleration_sample_KL(index_xAcceleration)
disp('对应的采样标准差：')
i = 0:0.01:0.4;
i(index_xAcceleration)

index_InitFollowingXVelocity = find(InitFollowingXVelocity_sample_KL == min(InitFollowingXVelocity_sample_KL));
disp('跟车初始速度的最小KL散度值为：')
InitFollowingXVelocity_sample_KL(index_InitFollowingXVelocity)
disp('对应的采样标准差：')
i = 0:0.01:6;
i(index_InitFollowingXVelocity)
%% 根据上述最优解，提取MH采样结果，并绘制直方图和概率密度
InitDhw_sample = InitDhw_sample_dataset(:,index_InitDhw);
dx_sample = dx_sample_dataset(:,index_dx);
xVelocity_sample = xVelocity_sample_dataset(:,index_xVelocity);
xAcceleration_sample = xAcceleration_sample_dataset(:,index_xAcceleration);
InitFollowingXVelocity_sample = InitFollowingXVelocity_sample_dataset(:,index_InitFollowingXVelocity);
% 得到直方图分布
[f_ecdf_InitDhw_sample, xc_ecdf_InitDhw_sample] = ecdf(InitDhw_sample);
[f_ecdf_dx_sample, xc_ecdf_dx_sample] = ecdf(dx_sample);
[f_ecdf_xVelocity_sample, xc_ecdf_xVelocity_sample] = ecdf(xVelocity_sample);
[f_ecdf_xAcceleration_sample, xc_ecdf_xAcceleration_sample] = ecdf(xAcceleration_sample);
[f_ecdf_InitFollowingXVelocity_sample, xc_ecdf_InitFollowingXVelocity_sample] = ecdf(InitFollowingXVelocity_sample);
% 得到核概率密度估计
pts = linspace(-78,-5,1000);
[f_ks_InitDhw_sample,xi_InitDhw_sample] = ksdensity(InitDhw_sample,pts);
pts = linspace(1,300,1000);
[f_ks_dx_sample,xi_dx_sample] = ksdensity(dx_sample,pts);
pts = linspace(2,55,1000);
[f_ks_xVelocity_sample,xi_xVelocity_sample] = ksdensity(xVelocity_sample,pts);
pts = linspace(-5,4,1000);
[f_ks_xAcceleration_sample,xi_xAcceleration_sample] = ksdensity(xAcceleration_sample,pts);
pts = linspace(4,58,1000);
[f_ks_InitFollowingXVelocity_sample,xi_InitFollowingXVelocity_sample] = ksdensity(InitFollowingXVelocity_sample,pts);
%% 最佳MH采样结果绘图
figure(1);
ecdfhist(f_ecdf_InitDhw_sample, xc_ecdf_InitDhw_sample,100);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
plot(xi_InitDhw_sample,f_ks_InitDhw_sample,'red','linewidth',3)
title('MH采样后初始纵向距离分布')
grid on;

figure(2);
ecdfhist(f_ecdf_dx_sample, xc_ecdf_dx_sample,100);
hold on;
xlabel('距离 (m/s)');  % 为X轴加标签
xlim([1 300]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dx_sample,f_ks_dx_sample,'red','linewidth',3)
title('MH采样后超车纵向行驶距离分布')
grid on;

figure(3);
ecdfhist(f_ecdf_xVelocity_sample, xc_ecdf_xVelocity_sample,100);
hold on;
xlabel('超车车速度 (m/s)');  % 为X轴加标签
xlim([2 55]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xVelocity_sample,f_ks_xVelocity_sample,'red','linewidth',3)
grid on;
title('MH采样后超车车速度分布'); 

figure(4);
ecdfhist(f_ecdf_xAcceleration_sample, xc_ecdf_xAcceleration_sample,100);
hold on;
xlabel('纵向加速度 (m/s^2)');  % 为X轴加标签
xlim([-5 4]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xAcceleration_sample,f_ks_xAcceleration_sample,'red','linewidth',3)
grid on;
title('MH采样后超车车纵向加速度分布')

figure(5);
ecdfhist(f_ecdf_InitFollowingXVelocity_sample, xc_ecdf_InitFollowingXVelocity_sample,100);
hold on;
xlabel('速度 (m/s)');  % 为X轴加标签
xlim([4 58]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_InitFollowingXVelocity_sample,f_ks_InitFollowingXVelocity_sample,'red','linewidth',3)
grid on;
title('MH采样后跟车初始纵向速度');
%% Overlay the theoretical density，将原始直方图、原始概率密度估计和MH采样的概率密度估计叠绘
figure(1);
ecdfhist(f_ecdf_InitDhw, xc_ecdf_InitDhw,100);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_InitDhw,f_ks_InitDhw,'red','linewidth',3)
hold on;
plot(xi_InitDhw_sample,f_ks_InitDhw_sample,'blue','linewidth',3)
title('初始相对距离分布对比')
grid on;

figure(2);
ecdfhist(f_ecdf_dx, xc_ecdf_dx,100);
hold on;
xlabel('距离 (m)');  % 为X轴加标签
xlim([1 300]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_dx,f_ks_dx,'red','linewidth',3)
hold on;
plot(xi_dx_sample,f_ks_dx_sample,'blue','linewidth',3)
title('超车距离分布对比')
grid on;

figure(3);
ecdfhist(f_ecdf_xVelocity, xc_ecdf_xVelocity,100);
hold on;
xlabel('速度 (m/s)');  % 为X轴加标签
xlim([2 55]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xVelocity,f_ks_xVelocity,'red','linewidth',3)
hold on;
plot(xi_xVelocity_sample,f_ks_xVelocity_sample,'blue','linewidth',3)
grid on;
title('超车车速度分布对比'); 

figure(4);
ecdfhist(f_ecdf_xAcceleration, xc_ecdf_xAcceleration,100);
hold on;
xlabel('加速度 (m/s^2)');  % 为X轴加标签
xlim([-5 4]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_xAcceleration,f_ks_xAcceleration,'red','linewidth',3)
hold on;
plot(xi_xAcceleration_sample,f_ks_xAcceleration_sample,'blue','linewidth',3)
grid on;
title('超车车加速度分布对比')

figure(5);
ecdfhist(f_ecdf_InitFollowingXVelocity, xc_ecdf_InitFollowingXVelocity,100);
hold on;
xlabel('速度 (m/s)');  % 为X轴加标签
xlim([4 58]);
ylabel('f(x)');  % 为Y轴加标签
% 绘制核密度估计图，并设置线条为黑色实线，线宽为3
plot(xi_InitFollowingXVelocity,f_ks_InitFollowingXVelocity,'red','linewidth',3)
hold on;
plot(xi_InitFollowingXVelocity_sample,f_ks_InitFollowingXVelocity_sample,'blue','linewidth',3)
grid on;
title('跟车自车初始速度分布对比');