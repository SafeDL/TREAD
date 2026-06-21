# SAIRL_subset：SAIRL 长尾安全评估子集

`SAIRL_subset/` 复制并改造 `IDM_subset/` 的 latent-space 子集模拟管线，用同一批 highD 长尾
scenario-condition 分布、同一扩散模型、同一 EVT 阈值和同一子集模拟超参数评估 SAIRL ego
策略。目标是只替换被测 ADS 的闭环响应，保持 adversary 轨迹生成和风险评分口径与 baseline
可比。

默认 SAIRL checkpoint：

```text
ref_code/Safe_imitation_learning-master/trained_models/SAIRL/model_1/195model.ckpt
```

运行时会优先加载已转换的 PyTorch/NPZ 权重：

```text
SAIRL_subset/results/policy_weights/sairl_195model_policy_net.npz
```

如果该文件不存在，`SAIRLPolicy` 会尝试用 TensorFlow 1.x compatibility API 从原 checkpoint
转换。当前 shell 环境若没有 TensorFlow，请在项目要求的环境中运行：

```bash
conda activate tread
```

也可以先显式转换：

```bash
python SAIRL_subset/scripts/convert_sairl_checkpoint.py
```

## 运行

following：

```bash
python SAIRL_subset/scripts/run_subset_following.py
```

cut-in：

```bash
python SAIRL_subset/scripts/run_subset_cutin.py
```

常用可比性覆盖项：

```bash
python SAIRL_subset/scripts/run_subset_following.py \
  --seed 42 \
  --num_samples 3000 \
  --p0 0.2 \
  --proposal_std 0.12
```

## 输出

结果写入：

```text
SAIRL_subset/results/following/sairl_following_result.json
SAIRL_subset/results/cutin/sairl_cutin_result.json
SAIRL_subset/results/SAIRL_evaluation_summary.json
SAIRL_subset/results/SAIRL_evaluation_summary.md
```

`sairl_*_result.json` 是压缩后的公开结果，只保留可比性判断需要的字段：

```text
ads
event_type
failure_event
fairness_checks
subset
monte_carlo
global_exposure
source_files
```

完整诊断仍保留在同目录的 `latent_subset_summary.json`、`latent_monte_carlo_summary.json`、
`latent_subset_level_stats.csv` 和样本 NPZ 中。若 mileage return-period 映射通过可靠性检查，
压缩结果会保留与 baseline 相同口径的 `global_exposure`，用于比较 SAIRL 与 highD 人类参考风险强度。

## 主要改动

- `src/sairl_policy.py`：加载 SAIRL checkpoint，定义 `SAIRLPolicy.reset()` 和
  `SAIRLPolicy.act(observation)`。
- `src/closed_loop_runner.py`：用 `SAIRLEgoVehicle` 替换 baseline IDM ego，adversary 扩散计划、
  EVT 风险评分和 trace 字段保持原管线。
- `scripts/convert_sairl_checkpoint.py`：将参考 TensorFlow checkpoint 转换为 PyTorch/NPZ 权重。
- `scripts/run_subset_following.py`、`scripts/run_subset_cutin.py`：分别运行 following/cut-in
  评估，并写出压缩后的 SAIRL 结果 JSON。
- `scripts/configs/latent_subset_*.yaml`：复用 baseline 输入路径，输出改到 `SAIRL_subset/results/`。
- `src/final_level_playback.py`：回放最终层样本时保留 ego/target，并过滤回放窗口内发生换道的其他
  highD 背景车，避免背景车与主车碰撞干扰 ADS 风险估计。
