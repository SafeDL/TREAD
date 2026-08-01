# cut-in：highD SS 参数敏感性

本子目录描述 highD cut-in 事件的冻结 OAT 设计；执行器位于父目录的
`run_experiments.py`，从而使两个事件共享同一运行留痕、失败判定和汇总规则。

固定参数为基础配置
`IDM_subset/scripts/configs/latent_subset_cutin.yaml` 的值：`N=1000`、
`p0=0.10`、`proposal_std=0.10`、`context_refresh_prob=0.50`、
`mh_retries_per_sample=4`、`max_levels=8`，且保留原有 adaptive stop 配置。
冻结 highD tail distribution、EVT 阈值、扩散 checkpoint 和 IDM policy 不变。

逐项变化：

| 参数 | OAT 值 |
| --- | --- |
| `num_samples` | 500, 1000, 2000 |
| `p0` | 0.05, 0.10, 0.20 |
| `proposal_std` | 0.05, 0.10, 0.20 |
| `context_refresh_prob` | 0.25, 0.50, 0.75, 0.90 |

结果写入 `IDM_subset/results/revision_highd_ss_sensitivity/runs/cutin/`；
20,000-sample MC 参考值写入同一结果树的 `references/cutin/`。

```bash
/home/hp/anaconda3/envs/tread/bin/python \
  IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --event cutin --run-reference-mc
```
