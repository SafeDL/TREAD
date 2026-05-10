%-----------自动化测试脚本，针对sceario+simulink并行仿真-----------% 
% 2022-03-31 初版
% 2022-07-19 增加碰撞时候的vf,vh记录
clc
clear all
%% 设置要仿真的参数,例如把10万个测试用例，切分成多个batch，每一次调用simmymodel时传入一个batch
Sample = IS_cutin2_sample; % 列：vf Initdhw tf
% vh = 34.56; % 聚类1的自车速度中位值
vh = 38; % 聚类2的自车速度中位值
batch_size = 200; % 1个batch包含的用例数量
%% Bus Creation
open_system('ACC_CutIn_Scenario.slx');
CutIn_scenario = drivingScenario;
CutIn_scenario.SampleTime = 0.01;
modelName = 'ACC_CutIn_Scenario';
blk = find_system(modelName,'System','driving.scenario.internal.ScenarioReader');
s = get_param(blk{1},'PortHandles');
get(s.Outport(1),'SignalHierarchy');
%% 选择调用并行仿真或者单次仿真
% 单次仿真
tic
simresult1 = RunSingleModel(vh, Sample,CutIn_scenario);
toc
%% 并行仿真,切分多个batch
tic
simresult = []; % 用于迭代存储结果
if mod(length(Sample),batch_size) == 0
    % Sample正好是interval的整数倍
    batch_length = floor(length(Sample)/batch_size);
    
    for i = 1:batch_length
        start_index = (i-1)*batch_size+1;
        end_index = i*batch_size;
        vf_sweep = Sample(start_index:end_index,1); % 切入车初始速度
        Initdhw_sweep = Sample(start_index:end_index,2); % 相对距离
        time_sweep = Sample(start_index:end_index,3); % 切入车换道时间
        
        %------------ 将所有场景合成一个数组，数组的每一个元素对应一个场景实例,并行使用---------%
        CutIn_scenario_sweep = [];
        for idx = 1:batch_size
            CutIn_scenario = drivingScenario;
            CutIn_scenario.SampleTime = 0.01;
            % 定义场景的道路信息
            roadCenters = [0,0;2000,0]; % 2000米长的直线道路
            laneSpecification = lanespec(2, 'Width', 3.75);
            road(CutIn_scenario, roadCenters, 'Lanes', laneSpecification, 'Name', 'Road');
            % Add the ego vehicle
            egoVehicle = vehicle(CutIn_scenario, ...
                'ClassID', 1, ...
                'Position', [0 -1.875 0], ...
                'Mesh', driving.scenario.carMesh,...
                'Name', 'egoVehicle');
            
            % Add the adversaryCar
            adversaryCar = vehicle(CutIn_scenario, ...
                'ClassID', 1, ...
                'Mesh', driving.scenario.carMesh,...
                'Name', 'adversaryCar');
            
            % 提取参数：切入车初速度，自车初速度，初始相对距离、换道时间
            vf = vf_sweep(idx); % 前车初始速度
            Initdhw = Initdhw_sweep(idx); % 换道起始点与本车的纵向距离
            time = time_sweep(idx); % 前车换道时间
            
            % Define the path and velocity of adversaryCar
            % 换道起始点坐标
            waypoint_x_1 = Initdhw + 4.7 + 1.2; % 1.2m为导入simulink的修正值，左车道换道起始点
            waypoint_y_1 = 1.875;
            % 换道起始相近点坐标
            waypoint_x_2 = Initdhw + 4.7 + 1.2 + 0.1;
            waypoint_y_2 = 1.875;
            % 换道终点坐标
            waypoint_x_3 = waypoint_x_2 + vf*time;
            waypoint_y_3 = - 1.875;
            % 换道终段相近点坐标
            waypoint_x_4 = waypoint_x_3 + 0.1;
            waypoint_y_4 = - 1.875;
            % 直线道路终点坐标
            waypoint_x_5 = 2000;
            waypoint_y_5 = - 1.875;
            % 合并道路
            waypoint_x = [waypoint_x_1,waypoint_x_2,waypoint_x_3,waypoint_x_4,waypoint_x_5]; % 需要加上最后一个点，否则相当于突然减速
            waypoint_y = [waypoint_y_1,waypoint_y_2,waypoint_y_3,waypoint_y_4,waypoint_y_5];
            waypoints = cat(2,waypoint_x',waypoint_y');
            trajectory(adversaryCar, waypoints, vf);
            % 将当前设计的场景存入CutIn_scenario_sweep
            CutIn_scenario_sweep = [CutIn_scenario_sweep,CutIn_scenario];
        end
        % 遍历CutIn_scenario_sweep，并赋以相应命名
        for idx = 1:batch_size
            str_var = 'CutIn_scenario_';
            varname = genvarname([str_var, num2str(idx)]);
            eval([varname '= CutIn_scenario_sweep(idx);']);
        end
        %--------------------------------%
        
        temp = RunParModel(vh,batch_size);
        simresult = [simresult;temp];
    end
else
    % Sample的长度是interval的整数倍+1
    vf_sweep = Sample(:,1); % 切入车初始速度
    Initdhw_sweep = Sample(:,2); % 相对距离
    time_sweep = Sample(:,3); % 切入车换道时间
    batch_size = length(vf_sweep);
    
    %------------ 将所有场景合成一个数组，数组的每一个元素对应一个场景实例,并行使用---------%
    CutIn_scenario_sweep = [];
    for idx = 1:batch_size
        CutIn_scenario = drivingScenario;
        CutIn_scenario.SampleTime = 0.01;
        % 定义场景的公共信息
        roadCenters = [0,0;2000,0]; % 2000米长的直线道路
        laneSpecification = lanespec(2, 'Width', 3.75);
        road(CutIn_scenario, roadCenters, 'Lanes', laneSpecification, 'Name', 'Road');
        % Add the ego vehicle
        egoVehicle = vehicle(CutIn_scenario, ...
            'ClassID', 1, ...
            'Position', [0 -1.875 0], ...
            'Mesh', driving.scenario.carMesh,...
            'Name', 'egoVehicle');
        
        % Add the adversaryCar
        adversaryCar = vehicle(CutIn_scenario, ...
            'ClassID', 1, ...
            'Mesh', driving.scenario.carMesh,...
            'Name', 'adversaryCar');
        
        % 提取参数：切入车初速度，自车初速度，初始相对距离、换道时间
        vf = vf_sweep(idx); % 前车初始速度
        Initdhw = Initdhw_sweep(idx); % 换道起始点与本车的纵向距离
        time = time_sweep(idx); % 前车换道时间
        
        % Define the path and velocity of adversaryCar
        % 换道起始点坐标
        waypoint_x_1 = Initdhw + 4.7 + 1.2; % 1.2m为导入simulink的修正值，左车道换道起始点
        waypoint_y_1 = 1.875;
        % 换道起始相近点坐标
        waypoint_x_2 = Initdhw + 4.7 + 1.2 + 0.1;
        waypoint_y_2 = 1.875;
        % 换道终点坐标
        waypoint_x_3 = waypoint_x_2 + vf*time;
        waypoint_y_3 = - 1.875;
        % 换道终段相近点坐标
        waypoint_x_4 = waypoint_x_3 + 0.1;
        waypoint_y_4 = - 1.875;
        % 直线道路终点坐标
        waypoint_x_5 = 2000;
        waypoint_y_5 = - 1.875;
        % 合并道路
        waypoint_x = [waypoint_x_1,waypoint_x_2,waypoint_x_3,waypoint_x_4,waypoint_x_5]; % 需要加上最后一个点，否则相当于突然减速
        waypoint_y = [waypoint_y_1,waypoint_y_2,waypoint_y_3,waypoint_y_4,waypoint_y_5];
        waypoints = cat(2,waypoint_x',waypoint_y');
        trajectory(adversaryCar, waypoints, vf);
        % 将当前设计的场景存入CutIn_scenario_sweep
        CutIn_scenario_sweep = [CutIn_scenario_sweep,CutIn_scenario];
    end
    % 遍历CutIn_scenario_sweep，并赋以相应命名
    for idx = 1:batch_size
        str_var = 'CutIn_scenario_';
        varname = genvarname([str_var, num2str(idx)]);
        eval([varname '= CutIn_scenario_sweep(idx);']);
    end
    %--------------------------------%
    
    temp = RunParModel(vh,batch_size);
    simresult = [simresult;temp];
    
end

% 保存数据
save('IS_cutin2_result.mat','simresult');
toc
%% -----------并行仿真子程序--------------
function simresult = RunParModel(vh,batch_size)
% 1) Load model
% open_system('ACC_CutIn_Scenario');
clc
% load_system('ACC_Evaluation_ADST');
model = 'ACC_CutIn_Scenario';
% 2) Set up the sweep parameters
numSims = batch_size; % 每个batch运行的测试规模
% 3) Create an array of SimulationInput objects and specify the sweep value for each simulation
simIn = repmat(Simulink.SimulationInput,[1 numSims]); % 自动并行运行

for idx = 1:numSims
    simIn(idx) = Simulink.SimulationInput(model); % 模型名称
    simIn(idx) = simIn(idx).setBlockParameter('ACC_CutIn_Scenario/Vehicle and Environment/Vehicle Dynamics/Bicycle Model - Force Input','xdot_o',num2str(vh));
    simIn(idx) = simIn(idx).setBlockParameter('ACC_CutIn_Scenario/Velocity_set','Value',num2str(vh*3.6));
    % 导入相应的CutIn scenario
    simIn(idx) = simIn(idx).setBlockParameter('ACC_CutIn_Scenario/Vehicle and Environment/Actors and Sensor Simulation/Scenario Reader','ScenarioVariableName',"CutIn_scenario_"+num2str(idx));
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
if newout.status == 1
    newout.vh = out.vh(end);
else
    newout.vh = nan;
end
end

end
%% --------------------Simulation Model-------------------%%
function simresult = RunSingleModel(vh,sample,CutIn_scenario)
result = zeros(1,5);
% 定义场景的公共信息
roadCenters = [0,0;2000,0]; % 2000米长的直线道路
laneSpecification = lanespec(2, 'Width', 3.75);
road(CutIn_scenario, roadCenters, 'Lanes', laneSpecification, 'Name', 'Road');
% Add the ego vehicle
egoVehicle = vehicle(CutIn_scenario, ...
    'ClassID', 1, ...
    'Position', [0 -1.875 0], ...
    'Mesh', driving.scenario.carMesh,...
    'Name', 'egoVehicle');

% Add the adversaryCar
adversaryCar = vehicle(CutIn_scenario, ...
    'ClassID', 1, ...
    'Mesh', driving.scenario.carMesh,...
    'Name', 'adversaryCar');
% 循环所有测试用例
for NrofRuns = 1:1 % 一共执行的仿真次数
    clc
    disp(['RunSimulation: ' num2str(NrofRuns)]);
    % 提取参数：自车初速度，前车初速度、加速度、换道时间，相对距离
    preVel = sample(NrofRuns,1); % 前车初始速度
    time = sample(NrofRuns,2); % 前车换道时间
    dhw = sample(NrofRuns,3); % 换道起始点与本车的纵向距离

    % 设置simulink变量：自车初速度，设定速度
    set_param('ACC_CutIn_Scenario/Vehicle and Environment/Vehicle Dynamics/Bicycle Model - Force Input','xdot_o',num2str(vh));
    set_param('ACC_CutIn_Scenario/Velocity_set','Value',num2str(vh*3.6));

    % Define the path and velocity of adversaryCar
    % 换道起始点坐标
    waypoint_x_1 = dhw + 4.7 + 1.2; % 1.2m为导入simulink的修正值，左车道换道起始点
    waypoint_y_1 = 1.875;
    % 换道起始相近点坐标
    waypoint_x_2 = dhw + 4.7 + 1.2 + 0.1;
    waypoint_y_2 = 1.875;
    % 换道终点坐标
    waypoint_x_3 = waypoint_x_2 + preVel*time;
    waypoint_y_3 = - 1.875;
    % 换道终段相近坐标
    waypoint_x_4 = waypoint_x_3 + 0.1;
    waypoint_y_4 = - 1.875;
    % 直线道路终点坐标
    waypoint_x_5 = 2000;
    waypoint_y_5 = - 1.875;
    % 合并道路
    waypoint_x = [waypoint_x_1,waypoint_x_2,waypoint_x_3,waypoint_x_4,waypoint_x_5]; % 需要加上最后一个点，否则相当于突然减速
    waypoint_y = [waypoint_y_1,waypoint_y_2,waypoint_y_3,waypoint_y_4,waypoint_y_5];
    waypoints = cat(2,waypoint_x',waypoint_y');
    trajectory(adversaryCar, waypoints, preVel);
    % 仿真
    sim('ACC_CutIn_Scenario.slx');
    % 记录变量
    result(NrofRuns,1) = max(ITTC);
    result(NrofRuns,2) = min(Range);
    result(NrofRuns,3) = status(end);
    result(NrofRuns,4) = vh(end);
    result(NrofRuns,5) = vdiff(end);
       
end
simresult = result;
end
