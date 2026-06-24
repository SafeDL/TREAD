# Zhao 2023 Dangerous Lane-change Reproduction on highD

本目录是对 `zhao_2023_dangerous_lane_change.pdf` 的 highD 复现实现。所有新增代码、配置和产物均保存在 `zhao_2023_highd_repro/` 内；根目录已有代码和历史结果只被读取作为参考。

## 论文方法理解

赵等 2023 的核心流程是数据-模型驱动的危险变道场景泛化：

1. 从自然驾驶数据中筛选紧急变道轨迹。
2. 将变道轨迹统一长度和起点，训练 BN-AM-SeqGAN 生成背景车变道轨迹。
3. 通过两车安全距离/碰撞约束，为每条背景车轨迹计算被测自动驾驶车辆初始状态。
4. 组合背景车轨迹和被测车初始状态，形成危险变道测试场景库。
5. 用生成轨迹分布、变道完成时间分布、TTC 等指标评价场景有效性。

本复现保留上述关键闭环，并按论文重新约束轨迹表示：

- 数据源从 NGSIM US101 改为 highD。
- 生成器学习对象为论文中的 20 点 `(y, vx)` 序列，而不是完整状态序列；`x` 由 `vx` 积分得到，`vy/ax/ay` 由导数重建。
- 主生成器复现 BN-AM-SeqGAN：将 `(y, vx)` 连续序列量化为 32 x 32 个坐标 token，用带 BatchNorm 的 LSTM 作为生成策略，用 attention-CNN-attention 判别器评分，并采用 “MLE 预训练 + 判别器交叉熵 + 逐前缀 Monte Carlo rollout 策略梯度奖励” 的训练流程。
- 额外保留 vanilla SeqGAN、RankGAN 和 PCA-Gaussian 作为 baseline；baseline 产物不参与默认危险场景和 IDM 评估。
- 采样后只做车辆运动学约束：横向位移单调、纵向位置单调、速度平滑，避免离散 token 造成的折返轨迹。
- 被测车初始状态按论文式 (18)-(20) 计算：由背景车平均速度、变道持续时间、`t2=0.2s`、`amax=6m/s^2` 和车长得到 `dmin`、`vav` 和横向安全距离 `dL`。

## highD 适配说明

论文按 NGSIM 将“紧急变道”定义为变道完成时间小于 2 s。当前默认配置严格使用这个阈值，并读取 highD 全 60 个 recording。按车道中心 10%-90% 横向进度识别出的完整切入只有 14 条，因此本复现的主要误差来源是 highD 严格样本量远小于论文 NGSIM 的 511 条，而不是额外放宽筛选条件。

## 文件结构

- `config/default.yaml`: 默认实验配置。
- `src/data.py`: highD 切入事件提取和轨迹数据集构建。
- `src/generator.py`: PCA-Gaussian baseline 和统一训练入口。
- `src/bn_am_seqgan.py`: BN-AM-SeqGAN 主复现，包含 BN-LSTM 生成器、注意力判别器和策略梯度训练。
- `src/paper_figures.py`: 复现论文图 8-15 的轨迹、速度、时间、RMSE、损失和 TTC 图。
- `src/scenario.py`: 基于式 (18)-(20) 的危险场景构造、highway-env 固定车道 IDM 闭环评估。
- `src/visualize.py`: highway-env 可导入检查和轨迹可视化。
- `scripts/run_all.py`: 端到端复现入口。
- `scripts/plot_paper_figures.py`: 单独生成论文式实验图。
- `results/`: 当前复现实验数据、模型和指标。
- `figures/highway_env_case_000.png`: 第 0 个 IDM 回放场景图。
- `figures/paper_like_*.png`: 对应论文图 8-15 的复现实验图。

## 当前复现结果

当前结果由默认配置生成：

- highD recording: `1-60`
- 提取 2 秒内紧急切入事件: `14`
- BN-AM-SeqGAN 生成候选轨迹: `50000`
- 构造危险场景: `500`
- highway-env IDM 评估场景: `200`

关键指标：

- 主生成器: `bn_am_seqgan`
- BN-AM-SeqGAN teacher-forcing NLL: `2.210`
- 逐前缀 Monte Carlo rollout 次数: `8`
- 变道持续时间分布 RMSE: `0.0168`
- 生成轨迹有效率: `100.0%`
- 横向位置 RMSE 中位数: `0.136 m`
- 纵向速度 RMSE 中位数: `2.044 m/s`
- 生成轨迹横向单调率: `1.000`
- 生成轨迹纵向单调率: `1.000`
- highway-env 场景切入终点贴近 ego 车道比例: `1.000`
- IDM ego 换道开关: `false`
- IDM ego 最大横向偏移: `0.000 m`
- IDM 闭环 `TTC < 1 s` 比例: `1.000`
- 碰撞比例: `0.785`
- 近碰撞比例: `1.000`
- 最小 TTC 中位数: `0.000 s`

指标文件：

- `results/data_summary.json`
- `results/bn_am_seqgan_stats.json`
- `results/generation_metrics.json`
- `results/seqgan_generation_metrics.json`
- `results/rankgan_generation_metrics.json`
- `results/scenario_summary.json`
- `results/idm_summary.json`
- `results/idm_metrics.csv`
- `results/paper_like_figure_metrics.json`
- `results/duration_distribution_table.csv`
- `results/output_effectiveness_table.csv`
- `results/pca_generation_metrics.json`

## 运行方式

使用项目指定环境：

```bash
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/run_all.py
```

分步运行：

```bash
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/extract_data.py
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/train_generator.py
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/generate_scenarios.py
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/evaluate_idm.py
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/visualize_case.py
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/plot_paper_figures.py
```

`visualize_case.py` 会显式导入本地 `/home/hp/TREAD/HighwayEnv`，并保存 highway-env IDM 回放图到 `figures/highway_env_case_000.png`。IDM 被测车使用 `tools/idm_ego.yaml` 的参数，但评估层强制 `enable_lane_change=false` 且 `steering=0`，因此 ego 只做纵向 IDM 加减速响应，不允许换道或横向漂移。

## 可调参数

常用参数都在 `config/default.yaml`：

- `data.recording_ids`: highD recording 范围。
- `data.max_lane_change_seconds`: 切入轨迹持续时间上限。
- `generator.method`: 默认主方法为 `bn_am_seqgan`。
- `generator.sample_count`: 生成轨迹数量。
- `seqgan.y_bins`, `seqgan.vx_bins`: `(y, vx)` 坐标量化粒度。
- `seqgan.pretrain_epochs`, `seqgan.adversarial_epochs`: BN-AM-SeqGAN 训练轮数。
- `seqgan.rollout_count`: 每个前缀的 Monte Carlo 补全次数。
- `seqgan.train_vanilla_baseline`: 是否训练 vanilla SeqGAN 对照。
- `seqgan.train_rankgan_baseline`: 是否训练 RankGAN 对照。
- `generator.pca_components`: PCA-Gaussian baseline 维度。
- `scenario.sample_count`: 危险场景数量。
- `simulation.max_scenarios`: IDM 闭环评估数量。

也可以复制一份配置后传入脚本：

```bash
/home/hp/anaconda3/envs/tread/bin/python zhao_2023_highd_repro/scripts/run_all.py zhao_2023_highd_repro/config/default.yaml
```

## BN-AM-SeqGAN 复现细节

论文的 BN-AM-SeqGAN 是离散序列 GAN：生成器输出下一坐标 token 的概率分布，判别器对完整轨迹序列给出真实性奖励。本实现按 highD 数据做了以下工程化对应：

- 轨迹 token: 每个 20 点轨迹只编码论文变量 `(y, vx)`，其中 `y` 是起点归一化后的横向位移，`vx` 是纵向速度。
- 生成器: `BNSeqGenerator` 使用 embedding、输入 BatchNorm、带门控 BatchNorm 的 LSTM cell 和 softmax 输出。
- 判别器: `AMDiscriminator` 使用 token embedding、序列自注意力、1D CNN、卷积后自注意力、池化和二分类层。
- 训练: 先用真实 token 序列最大似然预训练生成器，再用真实/生成序列交叉熵预训练判别器；对抗训练阶段对每个前缀 `L_{1:n}` 用当前 rollout policy `Gβ` 补全完整序列，经判别器 `Dφ` 评分后估计 `Q_D^G(L_{1:n-1}, l_n)`，再按 SeqGAN 策略梯度更新生成器，并在每轮后同步 `Gβ <- Gθ`。
- 场景构造: 生成轨迹转为完整 `[x, y, vx, vy, ax, ay]` 后，按论文式 (18)-(20) 计算 `dmin`、`vav` 和 `dL`。

`results/generated_trajectories.npz` 是 BN-AM-SeqGAN 主结果，并用于危险场景构造。SeqGAN、RankGAN 和 PCA-Gaussian 的大型 50k 轨迹 npz 是可再生成中间产物，当前目录只保留 baseline 模型、指标 json/csv 和图表，避免存放不必要的大文件。

## 对照实验

当前复现包含 3 个对抗生成对照：

- `SeqGAN`: LSTM 生成器 + CNN 判别器 + 逐前缀 Monte Carlo rollout。
- `RankGAN`: LSTM 生成器 + reference-aware ranking discriminator + 逐前缀 Monte Carlo rollout。
- `BN-AM-SeqGAN`: BN-LSTM 生成器 + attention-CNN-attention 判别器 + 逐前缀 Monte Carlo rollout。

`results/output_effectiveness_table.csv` 对应论文表 4。当前 3 个模型均生成 `50000` 条候选，并在 `duration<=2s`、相邻横向跳变不超过 `1.2m`、相邻纵向速度跳变不超过 `8m/s` 的条件下达到 `100.0%` 有效率。

## 论文式实验图

已生成的图表包括：

- `paper_like_fig08_trajectory_buffer.png`: 真实轨迹缓冲区和生成轨迹。
- `paper_like_fig09_start_speed_distribution.png`: 变道开始速度分布。
- `paper_like_fig10_end_speed_distribution.png`: 变道结束速度分布。
- `paper_like_table03_duration_distribution.png`: 变道完成时间分布。
- `paper_like_fig11_rmse_distribution.png`: 横向位置和纵向速度 RMSE。
- `paper_like_fig12_loss_curve.png`: RankGAN、SeqGAN 与 BN-AM-SeqGAN 损失曲线。
- `paper_like_fig13_14_dangerous_scenarios.png`: 危险变道场景轨迹。
- `paper_like_fig15_ttc_distribution.png`: TTC 分布。

## 局限

- 原论文使用 NGSIM US101 的 511 条紧急变道轨迹；这里严格使用 highD 全 60 个 recording 和 2 秒阈值，只提取到 14 条完整紧急切入轨迹。
- 在 14 条小样本上 vanilla SeqGAN 或 RankGAN 的 teacher-forcing NLL 可能低于 BN-AM-SeqGAN，这是过拟合风险，不代表论文结论在 highD 严格小样本上完全复现。
- highway-env 可视化中将论文坐标转换为“背景车从相邻车道切入 ego 车道”的 top-down 轨迹，IDM 动力学来自本地 highway-env。
