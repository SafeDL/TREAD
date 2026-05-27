# utils：跨模块公共工具

`utils/` 存放 TREAD 主线工程共同使用的轻量工具。凡是 `process_highD/`、
`diffusion/`、`adversaray/`、`subset/` 中出现重复实现，且语义不属于某个单独模块的
函数，都应优先放在这里。

## 当前内容

```text
utils/
├── io.py                 # resolve_path、load_npz、write_json、write_csv
├── evt.py                # POT/GPD EVT tail model、return level 和 S_EVT 标定
├── rss.py                # RSSConfig、安全距离、RSS margin、softmax pooling
├── risk.py               # y_long 计算、闭环风险和 EVT risk_score 标定
├── context.py            # context NPZ 读取和单条 context 组装
├── normalization.py      # numpy / torch 归一化与反归一化
└── diffusion_adapter.py  # frozen diffusion prior 的共享适配器
```

## 使用原则

- 风险评分统一从 `utils/risk.py` 引入，避免各工程维护不同公式。
- EVT 模型统一从 `utils/evt.py` 引入，避免 highD 拟合和闭环仿真使用不同尾部映射。
- RSS 基础计算统一从 `utils/rss.py` 引入，仅用于 KING 等显式 RSS 目标。
- NPZ、JSON、CSV 和配置路径解析统一使用 `utils/io.py`。
- 子模块不应新增仅做转发的兼容入口；调用点应直接 import `utils/` 中的真实实现。
- 不把模块私有训练逻辑、模型结构或脚本默认参数放进 `utils/`。

## 风险评分口径

`utils/risk.py` 提供同一套可配置的纵向风险评分实现：

- `process_highD/` EVT 拟合、长尾筛选、`adversaray/` 闭环验证和
  `subset/` 闭环仿真都先计算同一个 `y_long`：`1/TTC`、`1/THW`、
  `1/gap` 和 `DRAC` 的 softmax-pool 聚合，再加 collision、
  near collision 和 hard-brake 配置项。
- 如果配置 `evt.score_space: evt` 且提供 EVT model，`risk_score` 表示
  `S_EVT(y_long) = -log P_EVT(Y_long > y_long)`；否则 `risk_score`
  回退为 raw `y_long`。
- KING 优化阶段仍可使用 `risk_scoring` 中的 delta RSS 和 improper response；
  这些 adversarial RSS 项不在闭环验证或 highD tail 筛选中重新计算。

如果后续需要调整危险得分公式，应优先修改 `utils/risk.py` 和对应 YAML：
KING 优化使用 `risk_scoring`，闭环事件验证使用 `closed_loop_risk_scoring`。
修改后需要同步更新相关 README。

EVT 模型由 `process_highD/scripts/fit_longitudinal_evt.py` 生成，保存
`u, xi, beta, z20, z50, z100` 和 return level 置信区间。

共享长尾自然驾驶 context 由 `process_highD/scripts/select_tail_contexts.py` 生成。
这个筛选脚本默认使用 anchor 后到事件结束的逐帧 gap、TTC、THW 和 DRAC，
并调用同一个 RSS-free 纵向风险实现。
`adversaray/` 在这些长尾样本上做 KING 对抗优化，
`subset/` 在同一批长尾样本下继续做子集模拟。
