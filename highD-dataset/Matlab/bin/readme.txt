脚本
-------------------------------------------
KDF_proc.m  场景参数的核概率密度估计
LatFilter.m cut-in和超车等数据提取
LatinCube.m 拉丁抽样算法
LatOptimizeMH.m 对cut-in等参数的MH随机非均匀采样
LongFilter.m 纵向跟随数据提取
LongOptimizeMH.m  对跟驰工况的MH随机非均匀采样
OvertakeOptimizeMH.m 对超车等参数的MH随机非均匀采样
startVisualization.m 原始显示参数

函数
---------------------------------------------
KL_Calculate.m  KL散度计算
MH_Sampling.m 已知概率密度估计的MH采样
其余是拉丁超方抽样的函数（《代理模型的工程设计》）

MAT文件
----------------------------------------------------
Latdata.mat cut-in 所有换道数据预处理结果
Longdata.mat 所有跟随数据预处理结果
Overtakedata.mat   所有超车数据预处理结果

  
