"""
TREAD highD 第一阶段 — 尾部风险交互事件数据集构建器
==================================================
从 highD 原始自然驾驶轨迹数据中自动抽取 cut-in 与 car-following / hard-braking
交互事件，完成轨迹清洗、方向统一、ego-centric 坐标转换、TTC/THW/DRAC 风险指标
计算、固定长度窗口构建、风险分层标签生成，并导出可供 Deep EVT 与 diffusion 模型
训练的数据集。
"""

__version__ = "0.1.0"
