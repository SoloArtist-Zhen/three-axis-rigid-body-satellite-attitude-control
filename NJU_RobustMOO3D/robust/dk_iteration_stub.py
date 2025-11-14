"""
μ综合 (DK-iteration) 伪代码：
for it in range(max_iter):
    # D-scale identification：在频率网格上拟合对角/块对角 D(jw)
    D = fit_scaling(M_cl(jw))
    # H∞合成：在 D^{-1} M D^{-1} 上做一次 H∞（即上面的LMI或ARE近似）
    K = hinf_synthesis_weighted(D)
    # 检查 μ 上界或性能收敛；否则继续
"""
