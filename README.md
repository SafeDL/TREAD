# TREAD 项目说明

TREAD 面向 highD 自然驾驶数据中的跟驰与切入场景，构建从典型事件筛选、自然驾驶扩散先验、EVT 纵向风险标定到子集模拟失效概率估计的完整实验链路。

当前主线由三个工程和一个公共工具包组成：

```text
process_highD/   从 highD 原始轨迹中筛选典型驾驶事件
diffusion/       训练自然驾驶动作扩散先验
subset/          在 diffusion latent 空间中执行 subset simulation
utils/           跨工程共享的 IO、RSS、风险评分、context 和 diffusion adapter
```

## 统一口径

- `process_highD/` 只负责事件初筛和质量审计，不计算基础交互指标或综合危险得分。
- `diffusion/` 只学习自然驾驶动作分布，不把安全得分作为训练目标。
- `subset/` 的闭环事件风险统一输出为 `risk_score`，默认表示 EVT 标定后的 `S_EVT(y_long)`。
- highD EVT 建模先计算原始事件级纵向风险 `y_long`：`1/TTC`、`1/THW`、`1/gap` 和 `DRAC` 经 softmax pooling 聚合后，加 collision、near-collision 和 hard-brake 项。
- EVT 只定义自然驾驶纵向风险尾部分布和目标 return level；subset simulation 负责在闭环仿真分布中估计超过该等级的概率。
- highD EVT return level 是自然驾驶纵向风险尺度参考，不是 ADS collision probability，也不是直接的 human-vs-ADS crash-rate baseline。
- RSS 基础函数、归一化、context 读取、NPZ/JSON/CSV IO 等共性函数统一放在 `utils/`；RSS 不进入 EVT、tail context 筛选或 subset score。
- 结果文件鼓励简洁，只保存复现实验和分析需要的字段。

## 数据划分

`train`、`val`、`test` 按 recording id 划分，避免同一个 recording 的窗口同时出现在不同 split。
默认比例为 `0.70 / 0.15 / 0.15`，随机种子为 `42`。

- `process_highD/` 输出事件全集，不直接消费 split。
- `extract_highd_events.py` 同步缓存 following event 的 `y_long`、风险分量和
  `context_states`，供 EVT 拟合与 tail context 选择复用。
- `process_highD/scripts/build_natural_dataset.py` 构建 diffusion 数据集，并写入每个样本的 `split_index`。
- `diffusion/` 用 train 训练归一化统计和模型，用 val 做训练期间验证，用 test 做最终自然先验评估。
- `process_highD/scripts/select_tail_contexts.py` 默认按 EVT 拟合的 POT 阈值
  `u`，从全部有效 following events 中选择 `y_long > u` 的 highD tail
  contexts；当前为便于 subset 测试，默认从 tail 集合中按固定随机种子抽取
  `500` 个 context。若将 `num_contexts` 改为 `0`，则保留全部 tail contexts。
- `subset/` 默认消费这批 highD tail contexts，估计相对于该 context 分布的闭环极端风险概率。

## 安全得分公式

项目当前保留两类安全/风险计算：

1. 原始事件级纵向风险变量：

```text
longitudinal_proxy =
  w_ttc * softmax_pool(1/TTC)
+ w_thw * softmax_pool(1/THW)
+ w_gap * softmax_pool(1/gap)
+ w_drac * softmax_pool(DRAC)

y_long =
  longitudinal_proxy
+ collision_bonus * collision
+ near_collision_weight * near_collision
+ hard_brake_weight * hard_brake
```

该变量同时可从自然 highD 事件和 closed-loop trace 中计算；不使用 RSS，不做
duration/path-length normalization。

2. EVT 标定闭环验证/事件风险字段：

```text
risk_score = S_EVT(y_long)
           = -log P_EVT(Y_long > y_long)
```

EVT 模型由 `process_highD/scripts/fit_longitudinal_evt.py` 使用 highD following
events 的 `y_long` 通过 POT/GPD 拟合得到，输出 `u, xi, beta, z20, z50, z100`
和 return level 置信区间。`risk_score` 数值越大表示在 highD 自然纵向风险
尾部分布中越极端；它不是 ADS collision probability，也不能直接解释为
human baseline crash rate。

## 推荐运行顺序

所有命令默认从仓库根目录运行。项目默认环境为：

```bash
conda activate tread
```

1. 筛选 highD 事件：

```bash
python process_highD/scripts/extract_highd_events.py
```

2. 训练和评估自然驾驶扩散先验：

```bash
python process_highD/scripts/build_natural_dataset.py
python diffusion/scripts/train_natural_diffusion.py
python diffusion/scripts/evaluate_natural_prior.py
```

3. 拟合 highD 纵向风险 EVT，并选择共享长尾自然驾驶 contexts：

```bash
python process_highD/scripts/fit_longitudinal_evt.py
python process_highD/scripts/select_tail_contexts.py
```

4. 执行 latent-space subset simulation：

```bash
python subset/scripts/run_latent_subset_simulation.py
```

`pilot_subset_threshold.py` 仍可作为 pilot score diagnostic 脚本运行，但它不
定义最终失效等级；subset 主流程的最终 failure threshold 来自 EVT return
level，由 `subset/scripts/configs/latent_subset_simulation.yaml` 中的
`evt.return_period` 指定，当前默认是 `S_EVT(z100)`。

## 主要输出

```text
results/highd_events/                  事件表和质量报告
results/highd_events/following_event_* highD following 风险和 context 缓存
results/highd_evt/following/           highD 纵向风险 EVT 模型、诊断表和拟合图
results/highd_tail_contexts/following/ 共享长尾自然驾驶 contexts
results/diffusion_natural/following/   自然先验数据集、模型和评估结果
results/subset_simulation/             子集模拟结果、估计误差和诊断图
```

大型模型、数组、逐样本评分表和可再生成结果由 `.gitignore` 忽略；摘要 JSON、CSV 和诊断图可按需要保留。
