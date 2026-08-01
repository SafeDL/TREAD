# following：highD SS 参数敏感性

本子目录描述 highD 跟驰事件的冻结 OAT 设计；执行器位于父目录的
`run_experiments.py`，以避免跟驰与 cut-in 使用不同实现而产生估计器差异。

固定参数为基础配置
`IDM_subset/scripts/configs/latent_subset_following.yaml` 的值：`N=3000`、
`p0=0.20`、`proposal_std=0.12`、`context_refresh_prob=0.70`、
`mh_retries_per_sample=6`、`max_levels=8`，并保持冻结 highD tail distribution、
EVT 阈值、扩散 checkpoint 与 IDM policy 不变。

逐项变化：

| 参数 | OAT 值 |
| --- | --- |
| `num_samples` | 1000, 3000, 5000 |
| `p0` | 0.10, 0.20, 0.30 |
| `proposal_std` | 0.06, 0.12, 0.24 |
| `context_refresh_prob` | 0.30, 0.50, 0.70, 0.90 |

结果写入 `IDM_subset/results/revision_highd_ss_sensitivity/runs/following/`；
200,000-sample MC 参考值写入同一结果树的 `references/following/`。

```bash
/home/hp/anaconda3/envs/tread/bin/python \
  IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --event following --run-reference-mc
```
