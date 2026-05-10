%% 整理实际仿真的数据，得到ISO severity
%% 方法一，只是统计测试结果是否碰撞
clc
result = [];
parfor i = 1:100
    for j = 1:200
        if simresult(i,j).status == 1
            temp = 1;
            result = [result;temp];
        else
            temp = 3;
            result = [result;temp];
        end
    end
end
%% 补全cut-in 的vf数据, optional
parfor i = 1:100
    for j = 1:200
        if simresult(i,j).status == 1
            vh = simresult(i,j).vh;
            ittc = simresult(i,j).ITTC;
            range = simresult(i,j).Distance;
            vf = vh - ittc*range;
            simresult(i,j).vf = vf;
        else
            continue;
        end
    end
end
%% 方法二，将vh,vf,碰撞受伤程度等数据整理成多维数组
clc
a1 = -6.068;
b1 = -0.6234;
result = [];
parfor i = 1:100
    for j = 1:200
        if simresult(i,j).status == 1
            vh = simresult(i,j).vh;
            vf = simresult(i,j).vf;
            DeltaV = vh - vf;
            Y = 1/(1 + exp(- (a1 + 0.1*DeltaV*3.6 + b1)) );
            s_para = severity(DeltaV);
            temp = [vh,vf,DeltaV,Y,s_para];
            result = [result;temp];
        else
            continue;
        end
    end
end
disp("S0占比:")
length(find(result(:,5) == 0)) / length(result)
disp("S1占比:")
length(find(result(:,5) == 1)) / length(result)
disp("S2占比:")
length(find(result(:,5) == 2)) / length(result)
disp("S3占比:")
length(find(result(:,5) == 3)) / length(result)
%% 单独查看不同撞击速度下的受伤概率曲线
a1 = -6.068;
b1 = -0.6234;
DeltaV = 0:0.1:28; % (km/h)
Y = 1./(1 + exp(- (a1 + 0.1*DeltaV.*3.6 + b1)) );
plot(DeltaV,Y)
%% helper function 根据撞击速度，推测ISO 26262的severity
function s_para = severity(DeltaV)
DeltaV = DeltaV * 3.6;
if DeltaV <= 16
    s_para = 0;
elseif DeltaV <= 22
    s_para = 1;
elseif DeltaV <= 33
    s_para = 2;
else
    s_para = 3;
end
end





