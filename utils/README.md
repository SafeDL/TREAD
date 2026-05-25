# utils：跨模块公共工具

`utils/` 存放 TREAD 主线工程共同使用的轻量工具。凡是 `process_highD/`、
`diffusion/`、`adversaray/`、`subset/` 中出现重复实现，且语义不属于某个单独模块的
函数，都应优先放在这里。

## 当前内容

```text
utils/
├── io.py                 # resolve_path、load_npz、write_json、write_csv
├── rss.py                # RSSConfig、安全距离、RSS margin、softmax pooling
├── risk.py               # 统一风险评分、闭环风险、分位数排序工具
├── context.py            # context NPZ 读取和单条 context 组装
├── normalization.py      # numpy / torch 归一化与反归一化
└── diffusion_adapter.py  # frozen diffusion prior 的共享适配器
```

## 使用原则

- 风险评分统一从 `utils/risk.py` 引入，避免各工程维护不同公式。
- RSS 基础计算统一从 `utils/rss.py` 引入，不再在 `adversaray/` 和 `subset/` 中复制。
- NPZ、JSON、CSV 和配置路径解析统一使用 `utils/io.py`。
- 子模块不应新增仅做转发的兼容入口；调用点应直接 import `utils/` 中的真实实现。
- 不把模块私有训练逻辑、模型结构或脚本默认参数放进 `utils/`。

## 风险评分口径

`utils/risk.py` 提供同一套可配置的闭环评分实现，但不同模块使用不同权重：

- `adversaray/`：用于 KING before/after 对抗优化，默认包含相对 prior 的
  delta RSS、RSS improper response、TTC、DRAC、gap、碰撞和近碰撞。
- `subset/`：用于估计生成的 200 帧闭环事件的尾部概率，默认只把 collision、
  near collision、TTC、DRAC 和 gap 计入分数；RSS 量只保留为诊断，不进入分数。

如果后续需要调整危险得分公式，应优先修改 `utils/risk.py` 和对应 YAML：
KING 优化使用 `risk_scoring`，闭环事件验证使用 `closed_loop_risk_scoring`。
修改后需要同步更新相关 README。

共享长尾自然驾驶 context 由 `process_highD/scripts/select_tail_contexts.py` 生成。
这个筛选脚本默认使用 anchor 后到事件结束的逐帧 gap、TTC、THW、DRAC 和
closing speed 风险，并用 top-percentile mean 做长度相对聚合；前 50 帧
near-term 子分数只作为短期危险性的补充，不使用 RSS。
`adversaray/` 在这些长尾样本上做 KING 对抗优化，
`subset/` 在同一批长尾样本下继续做子集模拟。
