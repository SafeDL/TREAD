# Following 当前默认配置：5 次独立重复

本目录保存 IDM car-following 当前默认 SS 配置的 5 个预注册 seed：
`101, 202, 303, 404, 505`。当前 YAML 配置为：

```text
N=3000, p0=0.20, proposal_std=0.12, context_refresh_prob=0.70,
mh_retries_per_sample=6, max_levels=8, adaptive_stop_enabled=false
```

统一运行与续跑入口为：

```bash
python IDM_subset/experiments/highd_ss_sensitivity/run_experiments.py \
  --workflow default-repeats --event following
```

该工作流只运行 SS，顺序执行 5 个 seed，并读取
`../monte_carlo_following/latent_monte_carlo_summary.json` 作为 canonical 的 200,000-sample
MC 参考。运行器会在本根目录写入 `default_repeat_manifest.json`、`seed_results.csv` 与
`summary.json`，并拒绝复用操作性 SS 配置不一致的已有 seed。

当前保存的 5 个概率为 `0.001984`、`0.00334933`、`0.00259467`、`0.00237867`、`0.002160`；
均值为 `0.00249333`，跨 seed 标准差为 `0.00053073`。配对 MC 概率为 `0.00241000`。

这些 seed 目录来自先前冻结 OAT 默认单元的迁移，因此各 `effective_config.json` 中可能仍保留
旧的输出路径和 `sensitivity_experiment` metadata。统一运行器在复用时只校验实际 SS 配置；
不要手工改写这些快照。
