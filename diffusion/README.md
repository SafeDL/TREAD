# diffusion：自然驾驶动作先验

`diffusion/` 训练 highD 跟驰场景下的 lead vehicle 自然动作先验。它只建模“给定 ego-lead 历史，未来 lead 动作在自然数据中如何分布”，不负责风险筛选、EVT 标定或闭环安全概率估计。

默认口径：

```text
训练: DDPM noise prediction
推理: DDIM deterministic sampling
动作: jerk
```

## 运行顺序

```bash
conda run -n tread python process_highD/scripts/build_natural_dataset.py
conda run -n tread python diffusion/scripts/train_natural_diffusion.py
conda run -n tread python diffusion/scripts/evaluate_natural_prior.py
```

可选生成样本：

```bash
conda run -n tread python diffusion/scripts/sample_natural_rollouts.py
```

## 主要文件

```text
diffusion/scripts/configs/natural_following.yaml
diffusion/scripts/train_natural_diffusion.py
diffusion/scripts/evaluate_natural_prior.py
diffusion/scripts/sample_natural_rollouts.py
diffusion/src/data.py
diffusion/src/model.py
diffusion/src/train.py
diffusion/src/kinematics.py
```

## 主要输出

```text
results/diffusion_natural/following/dataset.npz
results/diffusion_natural/following/dataset_normalized.npz
results/diffusion_natural/following/feature_schema.json
results/diffusion_natural/following/normalization_stats.json
results/diffusion_natural/following/train_val_test_split.json
results/diffusion_natural/following/checkpoints/best_noise_mse.pt
results/diffusion_natural/following/training_summary.json
results/diffusion_natural/following/naturalness_summary.json
```

`dataset_normalized.npz` 只保存训练必需数组；`dataset.npz` 保存评估和样本回放需要的原始 reference。`feature_schema.json`、`normalization_stats.json` 和 `best_noise_mse.pt` 是 subset simulation 的必要输入。

## 与 subset 的关系

DDIM deterministic sampler 保证：

```text
same context + same latent z -> same action trajectory
```

因此 `subset/` 可以直接把 diffusion latent 作为随机变量，执行 latent-space subset simulation：

```text
context ~ Uniform(tail contexts)
z ~ N(0, I)
score = S_EVT(Y_long_sim)
```
