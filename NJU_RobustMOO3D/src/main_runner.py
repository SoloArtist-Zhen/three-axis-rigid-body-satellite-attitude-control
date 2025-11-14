from pathlib import Path
import glob

from .experiments import build_context, run_all_mc, run_quaternion
from .plotting import generate_all_figures, build_mosaics_by_type


def run_all(root: Path):
    """总调度：建模 -> Monte Carlo -> 四元数仿真 -> 画图 -> 3×3 拼图"""
    data_dir = root / "data"
    res_dir = root / "results"
    fig_dir = root / "figures"

    for d in [data_dir, res_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 搭建模型 + 控制器 + 不确定性 & 调度
    ctx = build_context(data_dir)

    # 2. 四种控制器的 Monte Carlo 指标
    dfs = run_all_mc(ctx, res_dir)

    # 3. 非线性四元数仿真数据
    quat = run_quaternion(ctx)

    # 4. 生成所有单图（包含箱线图 / 直方图 / 散点 / 热力图 / 时间响应 / 极点云）
    generate_all_figures(ctx, dfs, quat, fig_dir)

    # 5. 按类型做 3×3 拼图（box/hist/scat/heat/line/pole）
    build_mosaics_by_type(fig_dir)

    # 简单汇总
    single_pngs = list(fig_dir.glob("*.png"))
    mosaic_pngs = list(fig_dir.glob("mosaic_*_3x3.png"))
    n_mosaic = len(mosaic_pngs)
    n_single = len(single_pngs) - n_mosaic
    print(f"[INFO] Done. Single figures: {n_single}, mosaics: {n_mosaic}.")
