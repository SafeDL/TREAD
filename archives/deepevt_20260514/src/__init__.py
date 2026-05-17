"""DeepEVT 源码包。

为避免单元测试在 collection 阶段被强制加载 ``window_rebuild`` 等重模块
(后者依赖 ``tread_highd``),此处不做 eager import。
"""
