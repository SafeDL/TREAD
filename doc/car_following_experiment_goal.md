# Car-following 逻辑场景实验目标文档

代码运行在系统的conda环境：

```bash
conda activate tread
```


本文档用于指导 Codex 在当前仓库中补齐
car-following 逻辑场景的论文实验产物。本文档参考
`cutin\_experiment\_goal.md` 的组织方式，但所有路径、变量、指标和
概率语义均已改写为 following 场景，不应直接复用 cut-in 的 lateral
motion 或 cut-in semantic gate 字段。

任务目标不是重构方法、重训模型或重新运行完整 subset simulation，
而是基于当前已有代码、JSON/CSV/NPZ/PNG 结果，整理并补齐论文实验
部分所需的表格、图和诊断报告。

\---

## 0\. 总体约束

### 0.1 不破坏当前工程结构

1. 不移动、不重命名、不删除现有目录和结果文件。
2. 不修改已有模型 checkpoint、训练配置、核心训练和测试代码、
EVT 模型文件和 subset simulation 原始输出。
3. 所有新增实验汇总产物统一写入：

```text
results/paper\_experiments/following/
```

建议目录结构为：

```text
results/paper\_experiments/following/
├── tables/
├── figures/
├── logs/
├── cache/
├── following\_experiment\_manifest.json
└── FOLLOWING\_EXPERIMENT\_README.md
```

4. 现有结果目录只作为输入读取，包括但不限于：

```text
results/diffusion\_natural/following/
results/highd\_following\_tail/
results/subset\_simulation\_following/
results/highd\_events/
```

5. 如果某个目标图或目标表已经在已有结果目录中完成，则不要重复生成；
只需在 manifest 中登记其原始路径，并在
`FOLLOWING\_EXPERIMENT\_README.md` 中说明
“reused existing artifact”。

### 0.2 绘图风格保持全工程一致

1. 优先复用当前工程已有绘图函数、figure size、dpi、字体、线宽和命名风格。
2. 若没有统一绘图工具，则使用项目中已有 subset histogram 的默认风格。
3. 不引入 seaborn 或新的绘图主题。
4. 不改变已有图的视觉风格；新增图应采用简洁的 matplotlib 风格。
5. 建议保存为 `.png`，同时在条件允许时保存 `.pdf` 版本供论文使用。
6. 图中标签统一使用英文，例如：

   * `Risk score`
   * `Survival probability`
   * `EVT severity`
   * `ADS`
   * `highD human baseline`
   * `Car-following`
   * `Gap`
   * `TTC`
   * `THW`

### 0.3 跳过规则

每个实验在执行前都应检查目标产物是否已存在：

```text
if output exists and --force is not set:
    reuse output
else:
    generate output
```

如果输入文件不存在，不要伪造结果。应在 manifest 中记录：

```json
{
  "status": "skipped",
  "reason": "missing input file: ..."
}
```

\---

## 1\. 已知当前 following 结果文件

当前 car-following 实验至少应优先读取以下文件：

```text
results/diffusion\_natural/following/naturalness\_summary.json
results/subset\_simulation\_following/latent\_subset\_summary.json
results/highd\_following\_tail/evt/longitudinal\_peak\_evt\_model.json
results/highd\_following\_tail/evt/longitudinal\_peak\_evt\_summary.json
results/highd\_following\_tail/exposure/highd\_exposure\_summary.json
results/highd\_events/following\_event\_scores.csv
```

其中，部分 CSV/NPZ 文件可能因为 `.gitignore` 或本地环境差异不存在。
Codex 应做存在性检查；缺失时只跳过对应图或样本级统计，不应伪造数据。

从当前已提交的 following subset summary 可以读到以下关键信息：

```text
event\_type = following
probability = 0.08507
probability\_ci95\_lower = 0.08437148748499687
probability\_ci95\_upper = 0.08576851251500314
relative\_standard\_error = 0.004189302177337339
probability\_target =
  P\_context,z(Y\_long\_sim > x\_c | o sampled from highD tail scenario-condition distribution)
probability\_estimate\_kind = standard\_subset\_estimate
strict\_probability\_interpretation = true

num\_samples = 10000
num\_levels = 2
p0 = 0.1
proposal\_std = 0.08
context\_refresh\_prob = 0.25
mh\_retries\_per\_sample = 2
closed\_loop\_evaluations = 23479
proposal\_evaluations = 13479
stored\_level\_samples = 20000

level0\_failure\_fraction = 0.0848
final\_failure\_fraction = 0.8507
stop\_reason = subset\_threshold\_reached\_failure\_threshold
execution\_mode = fixed\_horizon
episode\_steps = 125

reliability\_status = pass
unique\_contexts\_final = 1303
unique\_states\_final = 7802
largest\_context\_share = 0.001
largest\_state\_share = 0.0008
transition\_acceptance\_rate = 0.6802
```

following 的人类驾驶基线和里程解释包括：

```text
primary\_exposure\_label = following ego
exposure\_denominator = following\_ego\_miles
primary\_exposure\_miles = 14594.223894953471
primary\_exposure\_hours = 224.60882222222207
all\_vehicle\_miles = 27635.889666845622
all\_vehicle\_hours = 440.16879999999986
num\_independent\_tail\_peaks = 7546

tail\_peak\_rate\_per\_mile = 0.5170538738006704
tail\_peak\_rate\_per\_hour = 33.596187030152294

ads\_safety\_critical\_intensity\_per\_mile = 0.043985773044223034
ads\_safety\_critical\_return\_period\_miles = 22.734623738330253
ads\_safety\_critical\_intensity\_per\_hour = 2.858027630655056
ads\_safety\_critical\_return\_period\_hours = 0.34989164879795137

highd\_safety\_critical\_intensity\_per\_mile = 0.008299226168158359
highd\_safety\_critical\_return\_period\_miles = 120.49316161989898
highd\_safety\_critical\_intensity\_per\_hour = 0.5392520367393501
highd\_safety\_critical\_return\_period\_hours = 1.8544204414073535

ads\_to\_highd\_intensity\_ratio\_per\_mile = 5.299984860393762
ads\_return\_period\_over\_highd\_return\_period\_miles = 0.18867978425238463
```

EVT 模型关键信息包括：

```text
risk\_variable = Y\_long
collision\_critical\_level = 5.0
evt\_failure\_threshold = 6.405841513141467
evt\_model\_u = 1.687343641294891
evt\_model\_xi = 0.17245406678854452
evt\_model\_beta = 0.5497100004585925
evt\_model\_exceedance\_rate = 0.10291450158886024
score\_space = evt
```

following 自然扩散先验评估包括：

```text
split = test
num\_samples = 1000
num\_available\_split\_samples = 61474
sample\_selection = random\_without\_replacement
sampler = ddim
action\_representation = jerk

val\_loss = 0.269069017319257
val\_noise\_mse = 0.22089947525609518
val\_x0\_l1 = 0.410378772546634
val\_smooth = 0.71316645210534

ax\_wasserstein = 0.005473576883223374
ax\_ks = 0.023031999999999997
jerk\_wasserstein = 0.055005991237729784
jerk\_ks = 0.10024800000000006

lead\_speed\_wasserstein = 0.01755840493774414
lead\_speed\_ks = 0.0048080000000000345
gap\_wasserstein = 0.04751473942565918
gap\_ks = 0.0024080000000000212
ttc\_wasserstein = 1.1237080721473693
ttc\_ks = 0.004624000000000017
thw\_wasserstein = 0.0014965696874856948
thw\_ks = 0.0022560000000000358

action\_clip\_rate = 0.0
speed\_negative\_rate = 0.0
jerk\_violation\_rate = 2.4e-05
ax\_violation\_rate = 0.0
trajectory\_discontinuity\_rate = 0.0
```

\---

## 2\. 实验章节目标总览

本次只针对 car-following 逻辑场景补齐实验产物。论文实验结构保持三节：

```text
4.1 自然驾驶建模与风险标尺有效性
4.2 EVT-targeted subset simulation 的长尾测试效果
4.3 消融实验与诊断分析
```

每一节内部可以有多个实验目标，但 Codex 执行时应优先生成最能支撑论文主线的表格和图。

\---

# 4.1 自然驾驶建模与风险标尺有效性

这一节回答两个问题：

1. highD-derived car-following 测试空间是否具有自然驾驶统计特征？
2. EVT 是否能给出 car-following 人类驾驶长尾风险标尺？

\---

## 实验 1：following 事件抽取、风险变量与 exposure 统计

### Goal

构建 car-following 逻辑场景的基础统计表，说明本文使用的 following 事件、
独立 tail peaks、following ego exposure 和风险变量 `Y\_long` 的数据基础。

### Inputs

优先读取：

```text
results/highd\_events/following\_event\_scores.csv
results/highd\_following\_tail/exposure/highd\_exposure\_summary.json
results/subset\_simulation\_following/latent\_subset\_summary.json
```

若 `following\_event\_scores.csv` 不存在，则从 subset summary 和 exposure summary
中提取可用统计量，并将风险变量分布图标记为 skipped。

### Required outputs

```text
results/paper\_experiments/following/tables/exp1\_following\_event\_exposure\_stats.csv
results/paper\_experiments/following/tables/exp1\_following\_event\_exposure\_stats.md
results/paper\_experiments/following/figures/exp1\_following\_y\_long\_hist\_ccdf.png
```

如果风险分布图无法生成，应输出：

```text
results/paper\_experiments/following/logs/exp1\_skipped\_y\_distribution.json
```

### Table fields

至少包含：

```text
event\_type
num\_scored\_following\_events
num\_semantic\_following\_events
num\_independent\_tail\_peaks
primary\_exposure\_label
exposure\_denominator
primary\_exposure\_miles
primary\_exposure\_hours
all\_vehicle\_miles
all\_vehicle\_hours
following\_ego\_mile\_fraction\_of\_all\_vehicle\_miles
tail\_peak\_rate\_per\_mile
tail\_peak\_rate\_per\_hour
collision\_critical\_level
evt\_failure\_threshold
```

若 `following\_event\_scores.csv` 存在，额外加入：

```text
y\_long\_mean
y\_long\_std
y\_long\_p50
y\_long\_p90
y\_long\_p95
y\_long\_p99
y\_long\_max
empirical\_exceedance\_rate\_at\_xc
min\_gap\_p05
min\_ttc\_p05
near\_collision\_rate
collision\_rate
```

### Figure requirement

`exp1\_following\_y\_long\_hist\_ccdf.png` 建议为双面板：

1. 左图：`Y\_long` histogram；
2. 右图：empirical CCDF；
3. 用竖线标记 `collision\_critical\_level = x\_c`；
4. 如果 EVT threshold 是 score-space threshold，不要把它直接画在 raw `Y\_long` 轴上；
raw 风险轴上只画 `x\_c`。

### Acceptance criteria

1. 表格可直接复制到论文。
2. 所有数值来自已有结果文件。
3. 如果输入缺失，清楚记录 skipped reason。
4. 不改变 `results/highd\_events/` 或 `results/highd\_following\_tail/` 中的原始文件。
5. exposure denominator 必须标注为 `following\_ego\_miles`，不要误写为 `all\_vehicle\_miles`。

\---

## 实验 2：POT/GPD 极值建模与 highD 人类驾驶基线

### Goal

生成 following EVT 参数表和 tail survival 曲线，证明 `x\_c` 是 highD 人类驾驶
following 长尾风险标尺，而不是人工 TTC/THW/DRAC 阈值。

### Inputs

```text
results/highd\_following\_tail/evt/longitudinal\_peak\_evt\_model.json
results/highd\_following\_tail/evt/longitudinal\_peak\_evt\_summary.json
results/subset\_simulation\_following/latent\_subset\_summary.json
```

如果 EVT model 或 summary 不存在，可从 `latent\_subset\_summary.json` 中提取
`evt\_model\_u`、`evt\_model\_xi`、`evt\_model\_beta`、`evt\_model\_exceedance\_rate` 等字段。

### Required outputs

```text
results/paper\_experiments/following/tables/exp2\_following\_evt\_params.csv
results/paper\_experiments/following/tables/exp2\_following\_evt\_params.md
results/paper\_experiments/following/figures/exp2\_following\_evt\_survival\_curve.png
results/paper\_experiments/following/figures/exp2\_following\_evt\_return\_level\_curve.png
```

若 survival curve 或 return-level curve 缺少必要输入，则跳过对应图，但仍生成参数表。

### Mathematical definition to use

对超过 POT 阈值 (u) 的风险值 (y>u)，GPD tail survival 写作：

\[
\\Pr(Y>y)
===

\\lambda\_u
\\left(
1+\\xi\\frac{y-u}{\\beta}
\\right)^{-1/\\xi},
]

其中：

\[
\\lambda\_u=\\Pr(Y>u)
]

是 exceedance rate。

EVT severity 定义为：

\[
S\_{\\mathrm{EVT}}(y)
===

\-\\log \\Pr(Y>y).
]

following 目标 failure event 写作：

\[
F\_x =
\\left{
Y^{ADS}*{\\mathrm{long}}>x\_c
\\right}
\\Longleftrightarrow
\\left{
S*{\\mathrm{EVT}}(Y^{ADS}\_{\\mathrm{long}})

>

S\_{\\mathrm{EVT}}(x\_c)
\\right}.
]

### Table fields

至少包含：

```text
event\_type
risk\_variable
pot\_threshold\_u
shape\_xi
scale\_beta
exceedance\_rate
collision\_critical\_level\_xc
evt\_failure\_threshold
score\_space
human\_safety\_critical\_intensity\_per\_mile
human\_safety\_critical\_return\_period\_miles
human\_safety\_critical\_intensity\_per\_hour
human\_safety\_critical\_return\_period\_hours
```

当前 following 结果应提取：

```text
risk\_variable = Y\_long
pot\_threshold\_u = 1.687343641294891
shape\_xi = 0.17245406678854452
scale\_beta = 0.5497100004585925
exceedance\_rate = 0.10291450158886024
collision\_critical\_level\_xc = 5.0
evt\_failure\_threshold = 6.405841513141467
```

### Figure requirements

#### Survival curve

横轴为 raw risk value (y)，纵轴为 (\\Pr(Y>y))，建议使用 log y-scale。
标记：

```text
POT threshold u
collision-critical level x\_c
```

#### Return-level curve

横轴为 return period 或 return distance，纵轴为 return level。
若当前工程已有类似图，则直接复用，不重复生成。

### Acceptance criteria

1. EVT 参数与 subset summary 中的数值一致。
2. 图中不要混淆 raw risk space 和 EVT score space。
3. 参数表必须包含 highD 人类驾驶基线强度和回报周期。
4. 不要将 TTC/THW/DRAC 经验阈值写成人类驾驶 calibrated return period。

\---

## 实验 3：following 扩散先验自然性验证

### Goal

验证 following diffusion prior 不是对抗式风险最大化器，而是保持 highD car-following
统计特征的自然前车动作生成模型。当前 following 模型采用 jerk action representation，
即生成前车纵向 jerk 序列，再通过闭环仿真中的动力学约束转化为前车纵向运动。

### Inputs

```text
results/diffusion\_natural/following/naturalness\_summary.json
```

可选输入：

```text
results/diffusion\_natural/following/natural\_prior\_plots/
results/diffusion\_natural/following/\*.npz
```

如果已有自然性图，则优先复用。

### Required outputs

```text
results/paper\_experiments/following/tables/exp3\_following\_naturalness\_summary.csv
results/paper\_experiments/following/tables/exp3\_following\_naturalness\_summary.md
results/paper\_experiments/following/figures/exp3\_following\_action\_distribution.png
results/paper\_experiments/following/figures/exp3\_following\_interaction\_naturalness.png
```

如果只有 `naturalness\_summary.json` 而没有原始样本数组，则只生成表格，不强行重画分布图。

### Table fields

建议压缩为以下指标：

```text
event\_type
split
num\_samples
num\_available\_split\_samples
sample\_selection
sampler
action\_representation
val\_loss
val\_noise\_mse
val\_x0\_l1
val\_smooth
ax\_wasserstein
ax\_ks
jerk\_wasserstein
jerk\_ks
lead\_speed\_wasserstein
lead\_speed\_ks
lead\_final\_speed\_wasserstein
lead\_final\_speed\_ks
lead\_displacement\_wasserstein
lead\_displacement\_ks
gap\_wasserstein
gap\_ks
ttc\_wasserstein
ttc\_ks
thw\_wasserstein
thw\_ks
relative\_speed\_wasserstein
relative\_speed\_ks
min\_gap\_wasserstein
min\_gap\_ks
min\_ttc\_wasserstein
min\_ttc\_ks
action\_clip\_rate
speed\_negative\_rate
jerk\_violation\_rate
ax\_violation\_rate
trajectory\_discontinuity\_rate
```

### Figure requirements

若能访问真实/生成样本数组，生成两张图：

1. `exp3\_following\_action\_distribution.png`

   * panels: acceleration (a\_x), jerk；
2. `exp3\_following\_interaction\_naturalness.png`

   * panels: gap, TTC, THW, relative speed 或 min gap。

若已有同类图，则复用已有图并在 manifest 中登记。

### Acceptance criteria

1. 表格反映 following 自然性验证的核心结果。
2. 图只在有原始分布数据时生成。
3. 不能仅为了画图而重新训练 diffusion model。
4. 不能把自然性验证写成风险最大化效果。
5. 不要加入 cut-in 专属指标，例如 lateral offset、lateral acceleration、
lateral jerk、cut-in semantic gate。

\---

# 4.2 EVT-targeted subset simulation 的 following 长尾测试效果

这一节回答：

在 highD-derived following tail scenario-condition distribution 中，
ADS 闭环响应超过 highD 人类驾驶 EVT safety-critical level 的概率是多少？

\---

## 实验 4：following 闭环长尾风险概率估计

### Goal

生成 following subset simulation 的核心结果表，作为论文主实验。

目标概率定义为：

\[
\\hat{p}\_{\\mathrm{following}}
===

\\Pr\_{o,z}
\\left\[
Y\_{\\mathrm{long,sim}}^{ADS}>x\_c
\\mid
o\\sim\\mathcal{D}\_{\\mathrm{tail}}^{H},
z\\sim\\mathcal{N}(0,I)
\\right].
]

### Inputs

```text
results/subset\_simulation\_following/latent\_subset\_summary.json
results/subset\_simulation\_following/latent\_subset\_level\_stats.csv
results/subset\_simulation\_following/latent\_subset\_samples.npz
results/subset\_simulation\_following/figures/subset\_score\_histograms.png
```

其中 `latent\_subset\_level\_stats.csv` 和 `latent\_subset\_samples.npz` 可能被 `.gitignore` 忽略。
若缺失，应从 summary 中读取 `level\_stats` 字段。

### Required outputs

```text
results/paper\_experiments/following/tables/exp4\_following\_subset\_main\_results.csv
results/paper\_experiments/following/tables/exp4\_following\_subset\_main\_results.md
results/paper\_experiments/following/figures/exp4\_following\_subset\_score\_histograms.png
results/paper\_experiments/following/figures/exp4\_following\_level\_score\_shift.png
```

若 `results/subset\_simulation\_following/figures/subset\_score\_histograms.png` 已存在，
则直接复用为 `exp4\_following\_subset\_score\_histograms.png` 或在 manifest 中登记原图路径。

### Table fields

至少包含：

```text
event\_type
probability\_target
failure\_event
probability
probability\_ci95\_lower
probability\_ci95\_upper
relative\_standard\_error
probability\_estimate\_kind
strict\_probability\_interpretation
num\_samples
num\_levels
p0
proposal\_std
context\_refresh\_prob
mh\_retries\_per\_sample
closed\_loop\_evaluations
proposal\_evaluations
final\_failure\_fraction
stop\_reason
reliability\_status
```

建议填入当前已知值：

```text
probability = 0.08507
95% CI = \[0.08437148748499687, 0.08576851251500314]
relative\_standard\_error = 0.004189302177337339
final\_failure\_fraction = 0.8507
num\_samples = 10000
num\_levels = 2
p0 = 0.1
proposal\_std = 0.08
context\_refresh\_prob = 0.25
mh\_retries\_per\_sample = 2
closed\_loop\_evaluations = 23479
proposal\_evaluations = 13479
```

### Figure requirements

#### Score histogram

复用已有 subset score histogram。
若必须重画，则：

1. 横轴为 EVT score 或 risk score；
2. 纵轴为 count 或 density；
3. 同图显示 level 0 与 final level；
4. 画出 failure threshold；
5. 图例包含 level index。

#### Level score shift

可以用 boxplot 或 quantile line 展示 level 0 到 final level 的 score 分布迁移。
若 summary 中只有 `level\_stats`，可画：

```text
score\_p50
score\_p90
score\_p95
score\_max
```

随 level 的变化曲线。

当前 following level statistics 包括：

```text
level0:
  score\_p50 = 5.141977299900391
  score\_p90 = 6.279306735852139
  score\_p95 = 6.911075199529104
  failure\_fraction = 0.0848

level1:
  score\_p50 = 6.893752530218456
  score\_p90 = 21.124078171039063
  score\_p95 = 23.72643892987287
  failure\_fraction = 0.8507
```

### Acceptance criteria

1. 主结果表可以直接放入论文。
2. 不重新运行 subset simulation。
3. 图和表必须说明 probability target 是 conditional on highD tail scenario-condition distribution。
4. 保留 `score\_space = evt` 的语义。

\---

## 实验 5：following ADS 与 highD 人类驾驶风险强度对比

### Goal

把 subset probability 转换为 safety-critical intensity 和 return period，
并与 highD 人类驾驶 baseline 对比。

### Inputs

```text
results/subset\_simulation\_following/latent\_subset\_summary.json
```

### Required outputs

```text
results/paper\_experiments/following/tables/exp5\_following\_ads\_vs\_highd\_intensity.csv
results/paper\_experiments/following/tables/exp5\_following\_ads\_vs\_highd\_intensity.md
results/paper\_experiments/following/figures/exp5\_following\_ads\_vs\_highd\_intensity.png
results/paper\_experiments/following/figures/exp5\_following\_ads\_vs\_highd\_return\_period.png
```

### Table fields

至少包含：

```text
event\_type
risk\_label
exposure\_denominator
tail\_peak\_rate\_per\_mile
ads\_exceedance\_probability\_conditional
ads\_safety\_critical\_intensity\_per\_mile
highd\_safety\_critical\_intensity\_per\_mile
ads\_to\_highd\_intensity\_ratio\_per\_mile
ads\_safety\_critical\_return\_period\_miles
highd\_safety\_critical\_return\_period\_miles
ads\_return\_period\_over\_highd\_return\_period\_miles
ads\_safety\_critical\_intensity\_per\_hour
highd\_safety\_critical\_intensity\_per\_hour
ads\_to\_highd\_intensity\_ratio\_per\_hour
ads\_safety\_critical\_return\_period\_hours
highd\_safety\_critical\_return\_period\_hours
following\_ego\_miles
following\_ego\_hours
all\_vehicle\_miles
all\_vehicle\_hours
```

当前 following 结果应得到：

```text
ADS intensity per mile = 0.043985773044223034
highD intensity per mile = 0.008299226168158359
ADS/highD intensity ratio = 5.299984860393762
ADS return period miles = 22.734623738330253
highD return period miles = 120.49316161989898
```

### Figure requirements

1. `exp5\_following\_ads\_vs\_highd\_intensity.png`

   * bar chart: ADS vs highD intensity per mile；
   * y-axis 建议使用 log scale；
2. `exp5\_following\_ads\_vs\_highd\_return\_period.png`

   * bar chart: ADS vs highD return period miles；
   * y-axis 建议使用 log scale。

### Acceptance criteria

1. 明确标注 denominator 为 `following\_ego\_miles`。
2. 不把该结果解释为无条件真实道路事故率。
3. 文本说明该结果是：
`conditional exceedance probability × highD tail peak exposure rate`。
4. 可以在附注中提供 all-vehicle background，但主结果应以 following ego miles 为主。

\---

# 4.3 消融实验与诊断分析

这一节回答：

1. subset simulation 是否比普通直接采样更有效？
2. EVT 风险标尺和 tail context distribution 是否必要？
3. final-level 结果是否存在样本坍缩或物理不可行？

如果部分消融缺少已有结果，不要强行运行昂贵流程。优先基于已有 summary 做轻量诊断。

\---

## 实验 6：采样策略消融

### Goal

比较 naive direct sampling 与 latent subset simulation 的估计效果。
如果当前仓库已有 risk-tilted sampling 或 empirical-tail-only 结果，则纳入对比；
否则在表中标记为 unavailable。

### Inputs

```text
results/subset\_simulation\_following/latent\_subset\_summary.json
results/subset\_simulation\_following/latent\_subset\_level\_stats.csv
results/\*\*/risk\_tilted\_\*
results/\*\*/empirical\*
```

### Required outputs

```text
results/paper\_experiments/following/tables/exp6\_following\_sampling\_ablation.csv
results/paper\_experiments/following/tables/exp6\_following\_sampling\_ablation.md
results/paper\_experiments/following/figures/exp6\_following\_sampling\_efficiency.png
```

### Baselines

#### Method A: Naive direct sampling

Use subset level 0 as direct Monte Carlo baseline:

```text
method = naive\_mc\_level0\_under\_tail\_context\_distribution
num\_samples = level\_stats\[0].num\_samples
failure\_fraction = level\_stats\[0].failure\_fraction
score\_mean = level\_stats\[0].score\_mean
score\_p90 = level\_stats\[0].score\_p90
score\_p95 = level\_stats\[0].score\_p95
score\_max = level\_stats\[0].score\_max
```

Current following values:

```text
num\_samples = 10000
failure\_fraction = 0.0848
score\_p90 = 6.279306735852139
score\_p95 = 6.911075199529104
score\_max = 27.631021115928547
```

#### Method B: Latent subset simulation

Use final subset result:

```text
method = latent\_subset
probability = 0.08507
final\_failure\_fraction = 0.8507
closed\_loop\_evaluations = 23479
proposal\_evaluations = 13479
num\_levels = 2
```

#### Method C: Risk-tilted sampling

Only include if existing result files are found.
Do not implement a new risk-tilted algorithm unless current code already contains it.

#### Method D: Empirical tail contexts only

Only include if existing result files are found.
Do not rerun full subset simulation unless an explicit config already exists.

### Figure requirement

`exp6\_following\_sampling\_efficiency.png` should compare:

```text
method
closed\_loop\_evaluations
estimated probability or failure fraction
relative standard error, if available
```

If only two methods are available, produce a two-row table and skip the figure if it would be misleading.

### Acceptance criteria

1. Do not pretend unavailable baselines exist.
2. Use level 0 as MC baseline only with clear label:
`naive\_mc\_level0\_under\_tail\_context\_distribution`.
3. Do not compare unconditional MC with tail-conditioned subset unless clearly normalized.
4. Do not infer unconditional road risk from tail-conditioned samples.

\---

## 实验 7：风险标尺消融

### Goal

比较 raw risk threshold、EVT severity threshold 和简单 surrogate threshold 的解释能力。
重点不是追求更多数值结果，而是证明 EVT 标尺能转换为 highD-calibrated return period。

### Inputs

优先使用：

```text
results/subset\_simulation\_following/latent\_subset\_samples.npz
results/subset\_simulation\_following/latent\_subset\_summary.json
results/highd\_events/following\_event\_scores.csv
```

若 `latent\_subset\_samples.npz` 不存在，则只生成定义性表格和 summary-based comparison。

### Required outputs

```text
results/paper\_experiments/following/tables/exp7\_following\_risk\_target\_ablation.csv
results/paper\_experiments/following/tables/exp7\_following\_risk\_target\_ablation.md
```

可选图：

```text
results/paper\_experiments/following/figures/exp7\_following\_risk\_target\_overlap.png
```

### Compared targets

至少定义以下三类：

```text
raw\_y\_long\_threshold:
    event = Y\_long\_sim > x\_c

evt\_severity\_threshold:
    event = S\_EVT(Y\_long\_sim) > S\_EVT(x\_c)

simple\_surrogate\_threshold:
    event = min\_ttc < tau\_ttc
            or min\_gap < tau\_gap
            or hard\_brake = 1
            or max\_drac > tau\_drac
```

若样本级 surrogate metrics 缺失，则 `simple\_surrogate\_threshold`
标记为 unavailable。

### Table fields

```text
target\_name
mathematical\_definition
requires\_evt\_model
can\_map\_to\_human\_return\_period
available\_in\_current\_outputs
estimated\_probability\_if\_available
ads\_highd\_intensity\_ratio\_if\_available
notes
```

### Acceptance criteria

1. 不需要强行运行新的风险评估。
2. 必须明确说明 TTC/THW/DRAC 阈值不能直接给出 highD-calibrated tail probability。
3. EVT severity 和 raw (Y>x\_c) 在单调 GPD score 下应保持等价排序，但解释空间不同。
4. following 场景不要使用 cut-in 专属 surrogate，如 lateral time gap 或 cut-in overlap。

\---

## 实验 8：context distribution 消融

### Goal

验证使用 highD following tail scenario-condition distribution 的必要性。
如果已有 alternative context results，则生成对比表；
否则生成待补实验清单，供后续单独运行。

### Inputs

搜索以下可能存在的结果：

```text
results/subset\_simulation\_following\*/latent\_subset\_summary.json
results/\*\*/following\*normal\*/latent\_subset\_summary.json
results/\*\*/following\*empirical\*/latent\_subset\_summary.json
results/\*\*/following\*copula\*/latent\_subset\_summary.json
```

### Required outputs

```text
results/paper\_experiments/following/tables/exp8\_following\_context\_distribution\_ablation.csv
results/paper\_experiments/following/tables/exp8\_following\_context\_distribution\_ablation.md
```

可选图：

```text
results/paper\_experiments/following/figures/exp8\_following\_context\_distribution\_ablation.png
```

### Compared context sources

```text
normal\_context\_distribution
empirical\_independent\_tail\_peaks
highd\_tail\_scenario\_condition\_distribution
```

Current main result should be labeled:

```text
context\_sampling\_mode = process\_highd\_tail\_distribution
probability\_target =
P\_context,z(Y\_long\_sim > x\_c | o sampled from highD tail scenario-condition distribution)
```

### Table fields

```text
context\_source
summary\_path
available
probability\_target
probability
ci95\_lower
ci95\_upper
num\_samples
num\_levels
unique\_contexts\_final
largest\_context\_share
strict\_probability\_interpretation
notes
```

### Acceptance criteria

1. Do not fabricate normal-context or empirical-only results.
2. If alternative results are unavailable, write `available = false` and explain what config/script is needed.
3. Do not modify existing subset configs while producing the paper artifacts.
4. Context source labels must use following terminology, not cut-in terminology.

\---

## 实验 9：可靠性、多样性与物理可行性诊断

### Goal

防止审稿人质疑 subset simulation 只是反复采样少数 context 或少数 latent state。
本实验生成 final-level 多样性和可靠性诊断表。

### Inputs

```text
results/subset\_simulation\_following/latent\_subset\_summary.json
results/diffusion\_natural/following/naturalness\_summary.json
results/subset\_simulation\_following/latent\_subset\_samples.npz
```

### Required outputs

```text
results/paper\_experiments/following/tables/exp9\_following\_reliability\_diagnostics.csv
results/paper\_experiments/following/tables/exp9\_following\_reliability\_diagnostics.md
results/paper\_experiments/following/figures/exp9\_following\_reliability\_diagnostics.png
```

### Table fields

```text
event\_type
reliability\_status
assessed\_level
acceptance\_rate
unique\_contexts
unique\_states
largest\_context\_share
largest\_state\_share
min\_unique\_contexts\_required
min\_unique\_states\_required
max\_largest\_context\_share\_allowed
max\_largest\_state\_share\_allowed
closed\_loop\_evaluations
stored\_level\_samples
physical\_feasibility\_proxy
action\_clip\_rate
speed\_negative\_rate
trajectory\_discontinuity\_rate
jerk\_violation\_rate
ax\_violation\_rate
```

Current expected values include:

```text
reliability\_status = pass
assessed\_level = 1
acceptance\_rate = 0.6802
unique\_contexts = 1303
unique\_states = 7802
largest\_context\_share = 0.001
largest\_state\_share = 0.0008
closed\_loop\_evaluations = 23479
stored\_level\_samples = 20000
```

### Figure requirement

A compact diagnostic bar chart is enough:

1. unique contexts vs required minimum；
2. unique states vs required minimum；
3. largest context share vs allowed maximum；
4. largest state share vs allowed maximum。

If the table is clear enough, figure can be skipped.

### Acceptance criteria

1. Reliability table must show `pass/fail/warning` status.
2. Diversity metrics must come from summary, not recomputed unless sample arrays exist.
3. Physical feasibility proxy can be taken from diffusion naturalness summary if subset-level
physical feasibility is unavailable.
4. Do not hide NaN final-level acceptance rate; report transition-level acceptance rate if available.
5. following 诊断不应出现 lateral acceleration、lateral jerk、cut-in overlap 等字段。

\---

# 5\. Unified manifest

Codex should create a manifest file:

```text
results/paper\_experiments/following/following\_experiment\_manifest.json
```

Suggested schema:

```json
{
  "experiment\_scope": "following",
  "created\_by": "tools/build\_following\_paper\_experiments.py",
  "source\_files": {
    "subset\_summary": "...",
    "naturalness\_summary": "...",
    "evt\_model": "...",
    "exposure\_summary": "..."
  },
  "experiments": {
    "exp1\_event\_exposure\_stats": {
      "status": "generated | reused | skipped",
      "outputs": \[],
      "skipped\_reason": null
    }
  }
}
```

Also create a human-readable report:

```text
results/paper\_experiments/following/FOLLOWING\_EXPERIMENT\_README.md
```

This README should list:

1. which artifacts were generated；
2. which artifacts were reused；
3. which experiments were skipped and why；
4. exact input file paths；
5. no-training / no-rerun statement；
6. whether exposure denominator is `following\_ego\_miles`；
7. whether any all-vehicle values are reported only as background.

\---

# 6\. 不建议 Codex 当前执行的任务

为避免破坏已有实验状态，Codex 当前不应执行：

1. 重新训练 following diffusion prior；
2. 重新拟合 following EVT 模型；
3. 重新运行完整 following subset simulation；
4. 修改 existing config 中的默认参数；
5. 删除或覆盖已有 `results/subset\_simulation\_following/` 中的文件；
6. 为了凑图而伪造缺失的样本级数据；
7. 将 following 结果解释为无条件真实道路事故率；
8. 把 all-vehicle exposure 作为 following 主 exposure denominator；
9. 把 cut-in 的 lateral 指标、semantic cut-in gate 或 cut-in overlap 复制到 following 实验中。

\---



