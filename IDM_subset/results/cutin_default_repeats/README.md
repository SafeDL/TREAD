# Cut-in 当前默认配置：5 次独立重复

本目录保存 IDM cut-in 当前默认 SS 配置的 5 个预注册 seed：
`101, 202, 303, 404, 505`。当前 YAML 配置为：

```text
N=2000, p0=0.05, proposal_std=0.10, context_refresh_prob=0.50,
mh_retries_per_sample=4, max_levels=8, adaptive_stop_enabled=false
```

统一运行与续跑入口为：

```bash
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow default-repeats --event cutin
```

该工作流只运行 SS，顺序执行 5 个 seed，并读取
`../monte_carlo_cutin/latent_monte_carlo_summary.json` 作为 canonical 的 20,000-sample MC
参考。运行器会使用或创建 `default_repeat_manifest.json`，并更新 `seed_results.csv` 与
`summary.json`；若当前 YAML 的操作性 SS 配置与已有 seed 不一致，会拒绝复用。

当前 5 个 SS 概率为 `0.005775`、`0.007700`、`0.007750`、`0.008600`、`0.003800`；均值为
`0.006725`，跨 seed 标准差为 `0.00193488`，95% t 区间为 `[0.004323, 0.009127]`。配对
20,000-sample MC 的概率为 `0.006950`，其 95% 区间为 `[0.005799, 0.008101]`；两区间重叠。

根目录已有的 `manifest.json` 来自已删除的历史校准入口，且其中记录的 10,000-sample MC
路径不是当前 canonical 参考。它只保留为历史追溯；统一运行器的 canonical manifest 是
`default_repeat_manifest.json`，当前 MC 以 `../monte_carlo_cutin/` 为准。
