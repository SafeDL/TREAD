%-----------自动化测试脚本，针对纯simulink并行仿真-----------% 
% 2022-03-27 初版
% 2022-07-19 为了分析事故发生时的危险性，增加了vh和vf的记录
clc
clear all
%% 设置要仿真的参数,例如把10万个测试用例，切分成多个batch，每一次调用simmymodel时传入一个batch
Sample = IS_following2_sample; % 列：vf af tf dhw
batch_size = 500; % 1个batch包含的用例数量
tao = 1.2; % ISO跟车距离的时距
d0 = 1.5; % ISO静止停车距离
preVel_sweep = Sample(:,1); % 前车初始速度
preA_sweep = Sample(:,2); % 前车加速度
time_sweep = Sample(:,3); % 前车变速时间   
Vend_sweep = preVel_sweep + preA_sweep.*time_sweep; % 前车变速终了速度
dhw_sweep = Sample(:,4); % 初始跟车距离  
% 找到前车终了减速至小于0的index
index1 = find((Vend_sweep < 0) & (preA_sweep < 0)); 
if ~isempty(index1)
    Vend_sweep(index1) = 0;
    time_sweep(index1) = abs(preVel_sweep(index1)./preA_sweep(index1));
end

% 找到前车终了加速至大于50的index，若测试用例不存在加速工况，可跳过
% index2 = find((Vend_sweep > 50) & (preA_sweep > 0)); 
% if ~isempty(index2)
%     Vend_sweep(index2) = 50;
%     time_sweep(index2) = abs((Vend_sweep(index2) - preVel_sweep(index2))./preA_sweep(index2));
% end

% 防止加减速时间为0，若测试用例不存在负的减速时间，可跳过
% index3 = find(time_sweep < 0);
% if ~isempty(index3)
%     time_sweep(index3) = 0.1;
% end

% 若测试用例指定了初始跟车距离，可跳过
% dhw_sweep =  tao.*preVel_sweep + d0; % 计算相对距离

df_sweep = dhw_sweep + 4.7; % 前车置心的初始位置=1.2+0.9+相对车间距离+1+1.6
%% 选择调用并行仿真或者单次仿真（并行调用入口）
% 单次仿真，重组sample
% tic
% sample = cat(2,preVel_sweep,preA_sweep,time_sweep,df_sweep);
% simresult = RunSingleModel(sample);
% toc

% 并行仿真,切分多个batch
tic
simresult = []; % 用于迭代存储结果
batch_length = floor(length(Sample)/batch_size);
for i = 1:batch_length
    start_index = (i-1)*batch_size+1;
    end_index = i*batch_size;
    temp = RunParModel(preVel_sweep(start_index:end_index),preA_sweep(start_index:end_index),time_sweep(start_index:end_index),df_sweep(start_index:end_index));
    simresult = [simresult;temp];
end
% Sample的长度是不是interval的整数倍，还要处理最后一段数据
if mod(length(Sample),batch_size) ~= 0
    start_index = end_index + 1;
    temp = RunParModel(preVel_sweep(start_index:end),preA_sweep(start_index:end),time_sweep(start_index:end),df_sweep(start_index:end));
    simresult = [simresult;temp];
end

% 保存数据
save('IS_following2_result.mat','simresult');
toc

% 并行仿真,不切分batch
% tic
% simresult = RunParModel(preVel_sweep,preA_sweep,time_sweep,df_sweep);
% toc
%% 并行仿真
function simresult = RunParModel(preVel_sweep,preA_sweep,time_sweep,df_sweep)
% 1) Load model
% open_system('ACC_Following_Scenario');
clc
% load_system('ACC_Following_Scenario');
model = 'ACC_Following_Scenario';
% 2) Set up the sweep parameters
numSims = length(preVel_sweep); % 每个batch运行的测试规模
% 3) Create an array of SimulationInput objects and specify the sweep value for each simulation
simIn = repmat(Simulink.SimulationInput,[1 numSims]); % 自动并行运行
for idx = 1:numSims
    simIn(idx) = Simulink.SimulationInput(model); % 模型名称
    simIn(idx) = simIn(idx).setBlockParameter('ACC_Following_Scenario/af','Value',num2str(preA_sweep(idx)));
    simIn(idx) = simIn(idx).setBlockParameter('ACC_Following_Scenario/time_switch','Threshold',num2str(time_sweep(idx)));
    simIn(idx) = simIn(idx).setBlockParameter('ACC_Following_Scenario/Lead Car/Bicycle Model - Force Input','X_o',num2str(df_sweep(idx)));
	simIn(idx) = simIn(idx).setBlockParameter('ACC_Following_Scenario/Lead Car/Bicycle Model - Force Input','xdot_o',num2str(preVel_sweep(idx)));
    simIn(idx) = simIn(idx).setBlockParameter('ACC_Following_Scenario/Ego Car/Bicycle Model - Force Input','xdot_o',num2str(preVel_sweep(idx)));

    simIn(idx) = simIn(idx).setPostSimFcn(@(x) postsim(x));
end
% 4) Simulate the model 
simresult = parsim(simIn,'ShowProgress', 'on','ShowSimulationManager','off','TransferBaseWorkspaceVariables','on');
% simresult = parsim(simIn);

% ---每次仿真结束后调用存储ITTC，distance，返回result值-----
function newout = postsim(out)
newout.ITTC = max(out.ITTC);
newout.Distance = min(out.Range);
newout.status = out.status(end);
% 如果发生碰撞记录vh,vf
if newout.status == 1
    newout.vf = out.vf(end);
    newout.vh = out.vh(end);
else
    newout.vf = nan;
    newout.vh = nan;
end

end
end
%% --------------------Simulation Model-------------------%%
function simresult = RunSingleModel(sample)
open_system('ACC_Following_Scenario.slx');
mdl = 'ACC_Following_Scenario.slx';
result = zeros(length(sample),6);
for NrofRuns = 1:length(sample) % 一共执行的仿真次数
    clc
    %设置此条件下的仿真环境目录
    disp(['RunSimulation: ' num2str(NrofRuns)]);
    % 提取参数：前车初速度、加速度、变速时间;计算前车末端速度，初始相对距离
    preVel = sample(NrofRuns,1); % 前车初始速度
    preA = sample(NrofRuns,2); % 前车加速度
    time = sample(NrofRuns,3); % 前车变速时间
    df = sample(NrofRuns,4); % 前车质心位置

    % 设置simulink变量
    set_param('ACC_Following_Scenario/af','Value',num2str(preA));
    set_param('ACC_Following_Scenario/time_switch','Threshold',num2str(time));
    set_param('ACC_Following_Scenario/Lead Car/Bicycle Model - Force Input','X_o',num2str(df));
	set_param('ACC_Following_Scenario/Lead Car/Bicycle Model - Force Input','xdot_o',num2str(preVel));
    set_param('ACC_Following_Scenario/Ego Car/Bicycle Model - Force Input','xdot_o',num2str(preVel));   
    sim(mdl);
    
    % 记录变量
    result(NrofRuns,1:3) = sample(NrofRuns,1:3);
    result(NrofRuns,4) = max(ITTC);
    result(NrofRuns,5) = min(Range);
    result(NrofRuns,6) = status(end);
end
simresult = result;
end

