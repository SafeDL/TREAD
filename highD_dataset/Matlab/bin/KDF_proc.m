%% 对提取的有效跟随数据中的场景参数做核密度估计
dhw = [];
xVelocity = [];
rVelocity = [];
xAcce = [];
TTC = [];
THW = [];
minVelocity = [];
maxVelocity = [];
meanVelocity = [];
minDHW = [];
minTHW = [];
minTTC = [];
preAx = [];
dectime = [];
vlead = [];
vend = [];
for i = 1:length(Result)
    dhw = [dhw;Result(i).dhw];% 所有跟随工况的跟车距离，m
    xVelocity = [xVelocity;Result(i).xVelocity];
    rVelocity = [rVelocity;Result(i).rVelocity;];% 所有跟随工况的相对速度,m/s
    xAcce = [xAcce;Result(i).xAcceleration];% 所有跟随工况的自车加速度
    THW = [THW;Result(i).THW]; % 所有跟随工况的跟车时距
    minVelocity = [minVelocity;Result(i).minVelocity]; % 所有跟随工况的自车最小速度
    maxVelocity = [maxVelocity;Result(i).maxVelocity]; % 所有跟随工况的自车最大速度
    meanVelocity = [meanVelocity;Result(i).meanVelocity]; % 所以跟随工况的自车平均速度
    minDHW = [minDHW;Result(i).minDHW]; % 所有跟随工况的最小跟车距离
    minTHW = [minTHW;Result(i).minTHW]; % 所有跟随工况的最小跟车时距
    preAx = [preAx;Result(i).preAx]; % 所有跟随工况的前车加速度
	minTTC = [minTTC;Result(i).minTTC]; % 所有跟随工况的最小TTC
	TTC = [TTC;Result(i).TTC]; % 所有跟随工况的TTC
    dectime = [dectime;Result(i).dectime]; % 所有跟随工况的减速时间
    vlead = abs([vlead;Result(i).vlead]);
    vend = abs([vend;Result(i).vend]);
end
% 剔除异常值
index = find(minTTC > 100);
minTTC(index) = [];
index = find(TTC > 100);
TTC(index) = [];
%% 对提取的跟随特征做箱型统计图
% 统计图
figure(1);    % 新建图形窗口
subplot(1,2,1)
boxlabel = {'跟车距离箱线图'};    % 箱线图的标签
% 绘制带有刻槽的箱线图
boxplot(dhw,boxlabel,'notch','on','orientation','vertical')
xlabel('dhw');  % 为X轴加标签
hold on
subplot(1,2,2)
boxlabel = {'主车速度箱线图'};    % 箱线图的标签
boxplot(xVelocity,boxlabel,'notch','on','orientation','vertical')
xlabel('xVelocity');  % 为X轴加标签

figure(2);
subplot(1,2,1)
boxlabel = {'相对速度箱线图'};    % 箱线图的标签
boxplot(rVelocity,boxlabel,'notch','on','orientation','vertical')
xlabel('rVelocity');  % 为X轴加标签
subplot(1,2,2)
boxlabel = {'主车加速度箱线图'};    % 箱线图的标签
boxplot(xAcce,boxlabel,'notch','on','orientation','vertical')
xlabel('xAcce');  % 为X轴加标签

figure(3);
subplot(1,2,1)
boxlabel = {'跟车时距箱线图'};    % 箱线图的标签
boxplot(THW,boxlabel,'notch','on','orientation','vertical')
xlabel('THW');  % 为X轴加标签
subplot(1,2,2)
boxlabel = {'跟车最小速度箱线图'};    % 箱线图的标签
boxplot(minVelocity,boxlabel,'notch','on','orientation','vertical')
xlabel('minXVelocity');  % 为X轴加标签

figure(4);
subplot(1,2,1)
boxlabel = {'跟车最大速度箱线图'};    % 箱线图的标签
boxplot(maxVelocity,boxlabel,'notch','on','orientation','vertical')
xlabel('maxXVelocity');  % 为X轴加标签
subplot(1,2,2)
boxlabel = {'跟车平均速度箱线图'};    % 箱线图的标签
boxplot(meanVelocity,boxlabel,'notch','on','orientation','vertical')
xlabel('meanXVelocity');  % 为X轴加标签

figure(5);
subplot(1,2,1)
boxlabel = {'跟车最小间距箱线图'};    % 箱线图的标签
boxplot(minDHW,boxlabel,'notch','on','orientation','vertical')
xlabel('minDHW');  % 为X轴加标签
subplot(1,2,2)
boxlabel = {'跟车最小时距箱线图'};    % 箱线图的标签
boxplot(minTHW,boxlabel,'notch','on','orientation','vertical')
xlabel('minTHW');  % 为X轴加标签

figure(6);
subplot(1,2,1)
boxlabel = {'前车加速度箱线图'};    % 箱线图的标签
boxplot(preAx,boxlabel,'notch','on','orientation','vertical')
xlabel('preAx');  % 为X轴加标签
subplot(1,2,2)
boxlabel = {'最小TTC箱线图'};    % 箱线图的标签
boxplot(minTTC,boxlabel,'notch','on','orientation','vertical')
xlabel('minTTC');  % 为X轴加标签

figure(7);
subplot(1,2,1)
boxlabel = {'TTC箱线图'};    % 箱线图的标签
boxplot(TTC,boxlabel,'notch','on','orientation','vertical')
xlabel('TTC');  % 为X轴加标签
subplot(1,2,2)
boxlabel = {'制动时间箱线图'};    % 箱线图的标签
boxplot(dectime,boxlabel,'notch','on','orientation','vertical')
xlabel('dectime');  % 为X轴加标签

figure(8);
subplot(1,2,1)
boxlabel = {'前车制动初始速度箱线图'};    % 箱线图的标签
boxplot(vlead,boxlabel,'notch','on','orientation','vertical')
xlabel('vlead');  % 为X轴加标签
subplot(1,2,2)
boxlabel = {'前车制动末端速度箱线图'};    % 箱线图的标签
boxplot(vend,boxlabel,'notch','on','orientation','vertical')
xlabel('vend');  % 为X轴加标签