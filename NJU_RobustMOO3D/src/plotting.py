from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .models import satellite_linear_model
from .controllers import simulate_linear, lqr, closed_loop, ise, energy, settling
from .metrics import spectral_abscissa, hinf_like
from compose_mosaic import stitch_3x3


# ---------- 基础工具：命名 + 保存 ----------
_fig_counters = defaultdict(int)


def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _newfig_name(fig_dir: Path, tag: str) -> Path:
    """
    tag: "box" | "hist" | "scat" | "heat" | "line" | "pole"
    """
    _fig_counters[tag] += 1
    return fig_dir / f"{tag}_{_fig_counters[tag]:02d}.png"


# 单类型底层接口
def _boxplot(fig_dir, datas, labels, title):
    plt.figure()
    plt.boxplot(datas, labels=labels, showmeans=True)
    plt.title(title)
    _savefig(_newfig_name(fig_dir, "box"))


def _hist(fig_dir, data, title, bins=20):
    plt.figure()
    plt.hist(data, bins=bins)
    plt.title(title)
    _savefig(_newfig_name(fig_dir, "hist"))


def _scatter(fig_dir, x, y, xl, yl, title):
    plt.figure()
    plt.scatter(x, y, s=14)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    _savefig(_newfig_name(fig_dir, "scat"))


def _heat(fig_dir, values, extent, title):
    plt.figure()
    plt.imshow(values, origin="lower", extent=extent, aspect="auto")
    plt.colorbar()
    plt.title(title)
    _savefig(_newfig_name(fig_dir, "heat"))


def _line_multi(fig_dir, Y, title):
    plt.figure()
    for k in range(min(5, Y.shape[1])):
        plt.plot(Y[:, k])
    plt.title(title)
    _savefig(_newfig_name(fig_dir, "line"))


def _pole_cloud(fig_dir, xs, ys, title):
    plt.figure()
    plt.scatter(xs, ys, s=8)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(title)
    _savefig(_newfig_name(fig_dir, "pole"))


# ---------- 1. 箱线图，补足 9 张 ----------
def _generate_boxplots(ctx, dfs, fig_dir: Path):
    df_lqr = dfs["lqr"]
    df_hinf = dfs["hinf"]
    df_mpc = dfs["mpc"]
    df_sched = dfs["sched"]

    # 1-5: 四控制器整体对比
    _boxplot(
        fig_dir,
        [df_lqr["ISE"], df_hinf["ISE"], df_mpc["ISE"], df_sched["ISE"]],
        ["LQR", "H∞", "tMPC", "Sched"],
        "ISE Distribution (All Controllers)",
    )
    _boxplot(
        fig_dir,
        [df_lqr["Eff"], df_hinf["Eff"], df_mpc["Eff"], df_sched["Eff"]],
        ["LQR", "H∞", "tMPC", "Sched"],
        "Energy Distribution (All Controllers)",
    )
    _boxplot(
        fig_dir,
        [df_lqr["Sett"], df_hinf["Sett"], df_mpc["Sett"], df_sched["Sett"]],
        ["LQR", "H∞", "tMPC", "Sched"],
        "Settling Time (All Controllers)",
    )
    _boxplot(
        fig_dir,
        [
            df_lqr["Hinf_like"],
            df_hinf["Hinf_like"],
            df_mpc["Hinf_like"],
            df_sched["Hinf_like"],
        ],
        ["LQR", "H∞", "tMPC", "Sched"],
        "H∞-like Gain (All Controllers)",
    )
    _boxplot(
        fig_dir,
        [
            df_lqr["SpecAbs"],
            df_hinf["SpecAbs"],
            df_mpc["SpecAbs"],
            df_sched["SpecAbs"],
        ],
        ["LQR", "H∞", "tMPC", "Sched"],
        "Spectral Abscissa (All Controllers)",
    )

    # 6-9: 配对对比
    _boxplot(
        fig_dir,
        [df_lqr["ISE"], df_hinf["ISE"]],
        ["LQR", "H∞"],
        "ISE (LQR vs H∞)",
    )
    _boxplot(
        fig_dir,
        [df_lqr["Eff"], df_hinf["Eff"]],
        ["LQR", "H∞"],
        "Energy (LQR vs H∞)",
    )
    _boxplot(
        fig_dir,
        [df_mpc["ISE"], df_sched["ISE"]],
        ["tMPC", "Sched"],
        "ISE (tMPC vs Sched)",
    )
    _boxplot(
        fig_dir,
        [df_mpc["Eff"], df_sched["Eff"]],
        ["tMPC", "Sched"],
        "Energy (tMPC vs Sched)",
    )


# ---------- 2. 直方图，补足 9 张 ----------
def _generate_histograms(ctx, dfs, fig_dir: Path):
    df_lqr = dfs["lqr"]
    df_hinf = dfs["hinf"]
    df_mpc = dfs["mpc"]
    df_sched = dfs["sched"]

    _hist(fig_dir, df_hinf["Hinf_like"], "H∞-like (H∞ Controller)")
    _hist(fig_dir, df_lqr["Hinf_like"], "H∞-like (LQR)")
    _hist(fig_dir, df_mpc["Hinf_like"], "H∞-like (tMPC)")
    _hist(fig_dir, df_sched["Hinf_like"], "H∞-like (Sched)")

    _hist(fig_dir, df_lqr["Eff"], "Energy (LQR)")
    _hist(fig_dir, df_hinf["Eff"], "Energy (H∞)")
    _hist(fig_dir, df_mpc["Eff"], "Energy (tMPC)")
    _hist(fig_dir, df_sched["Eff"], "Energy (Sched)")

    _hist(fig_dir, df_hinf["Sett"], "Settling Time (H∞)")


# ---------- 3. 散点图，补足 9 张 ----------
def _generate_scatters(ctx, dfs, fig_dir: Path):
    df_lqr = dfs["lqr"]
    df_hinf = dfs["hinf"]
    df_mpc = dfs["mpc"]
    df_sched = dfs["sched"]

    _scatter(fig_dir, df_lqr["ISE"], df_lqr["Eff"], "ISE", "Energy", "LQR ISE vs Energy")
    _scatter(fig_dir, df_hinf["ISE"], df_hinf["Eff"], "ISE", "Energy", "H∞ ISE vs Energy")
    _scatter(fig_dir, df_mpc["ISE"], df_mpc["Eff"], "ISE", "Energy", "tMPC ISE vs Energy")
    _scatter(fig_dir, df_sched["ISE"], df_sched["Eff"], "ISE", "Energy", "Sched ISE vs Energy")

    _scatter(
        fig_dir,
        df_hinf["Sett"],
        df_hinf["Hinf_like"],
        "Settling",
        "H∞-like",
        "H∞ Settling vs H∞-like",
    )
    _scatter(
        fig_dir,
        df_lqr["Sett"],
        df_lqr["Hinf_like"],
        "Settling",
        "H∞-like",
        "LQR Settling vs H∞-like",
    )
    _scatter(
        fig_dir,
        df_lqr["ISE"],
        df_lqr["Hinf_like"],
        "ISE",
        "H∞-like",
        "LQR ISE vs H∞-like",
    )
    _scatter(
        fig_dir,
        df_hinf["ISE"],
        df_hinf["Hinf_like"],
        "ISE",
        "H∞-like",
        "H∞ ISE vs H∞-like",
    )
    _scatter(
        fig_dir,
        df_mpc["Sett"],
        df_mpc["Hinf_like"],
        "Settling",
        "H∞-like",
        "tMPC Settling vs H∞-like",
    )


# ---------- 4. 热力图 / surface（含热力图），补足 9 张 ----------
def _generate_heatmaps(ctx, fig_dir: Path):
    """
    统一在这里生成 >=9 张热力图（heat_*.png），保证后面可以拼出 mosaic_heat_3x3.png
    """
    K_lqr = ctx["K_lqr"]
    K_hinf = ctx["K_hinf"]

    # ---------- 1. 惯量平面上的谱半径 / 稳定性映射 ----------
    grid = 18
    Jx = np.linspace(0.7, 1.3, grid)
    Jy = np.linspace(0.8, 1.4, grid)
    H_lqr = np.zeros((grid, grid))
    H_hinf = np.zeros((grid, grid))

    for a, x in enumerate(Jx):
        for b, y in enumerate(Jy):
            A, B, Bw, C, Du, Dw = satellite_linear_model(
                np.array([x, y, 1.0]), np.zeros((6, 6))
            )
            H_lqr[b, a] = spectral_abscissa(closed_loop(A, B, K_lqr))
            H_hinf[b, a] = spectral_abscissa(closed_loop(A, B, K_hinf))

    # 热图 1–3：LQR / H∞ / 差值
    _heat(fig_dir, H_lqr, [Jx[0], Jx[-1], Jy[0], Jy[-1]], "Spec Abscissa (LQR)")
    _heat(fig_dir, H_hinf, [Jx[0], Jx[-1], Jy[0], Jy[-1]], "Spec Abscissa (H∞)")
    _heat(
        fig_dir,
        H_hinf - H_lqr,
        [Jx[0], Jx[-1], Jy[0], Jy[-1]],
        "ΔSpec Abscissa (H∞-LQR)",
    )

    # ---------- 2. Resolvent norm（频域鲁棒性代理） ----------
    midw = 1.0
    Gmap_hinf = np.zeros((grid, grid))
    Gmap_lqr = np.zeros((grid, grid))
    for a, x in enumerate(Jx):
        for b, y in enumerate(Jy):
            A, B, Bw, C, Du, Dw = satellite_linear_model(
                np.array([x, y, 1.0]), np.zeros((6, 6))
            )
            Acl_h = closed_loop(A, B, K_hinf)
            Acl_l = closed_loop(A, B, K_lqr)
            G_h = np.linalg.solve(
                1j * midw * np.eye(Acl_h.shape[0]) - Acl_h, np.eye(Acl_h.shape[0])
            )
            G_l = np.linalg.solve(
                1j * midw * np.eye(Acl_l.shape[0]) - Acl_l, np.eye(Acl_l.shape[0])
            )
            Gmap_hinf[b, a] = np.linalg.svd(G_h, compute_uv=False).max()
            Gmap_lqr[b, a] = np.linalg.svd(G_l, compute_uv=False).max()

    # 热图 4–6：H∞ 的 resolvent，LQR 的 resolvent，二者差异
    _heat(fig_dir, Gmap_hinf, [Jx[0], Jx[-1], Jy[0], Jy[-1]], "Resolvent norm @ ω=1 (H∞)")
    _heat(fig_dir, Gmap_lqr, [Jx[0], Jx[-1], Jy[0], Jy[-1]], "Resolvent norm @ ω=1 (LQR)")
    _heat(
        fig_dir,
        Gmap_hinf - Gmap_lqr,
        [Jx[0], Jx[-1], Jy[0], Jy[-1]],
        "ΔResolvent norm (H∞-LQR)",
    )

    # ---------- 3. Q/R 权重扫描下的 ISE/Energy/Settling surface ----------
    A0 = ctx["A0"]
    B0 = ctx["B0"]
    Bw0 = ctx["Bw0"]

    qang = np.geomspace(0.02, 20.0, 8)
    rr = np.geomspace(0.05, 1.2, 8)
    Z_I = np.zeros((len(qang), len(rr)))
    Z_E = np.zeros_like(Z_I)
    Z_S = np.zeros_like(Z_I)

    from .controllers import simulate_linear, ise, energy, settling  # 局部导入防止循环

    for i, qa in enumerate(qang):
        for j, r in enumerate(rr):
            Qg = np.diag([qa, qa, qa, 1, 1, 1])
            Rg = np.eye(3) * r
            Ktmp, _ = lqr(A0, B0, Qg, Rg)
            xs, us = simulate_linear(
                A0,
                B0,
                Bw0,
                Ktmp,
                T=1.6,
                dt=0.01,
                disturb_scale=0.12,
                policy_type="statefb",
                u_limits=[-0.6, 0.6],
            )
            Z_I[i, j] = ise(xs, 0.01)
            Z_E[i, j] = energy(us, 0.01)
            Z_S[i, j] = settling(xs, 0.01)

    # 热图 7–9：ISE，Energy，Settling 的 surface
    _heat(fig_dir, Z_I, [0, 1, 0, 1], "ISE vs (q_angle,r) index surface")
    _heat(fig_dir, Z_E, [0, 1, 0, 1], "Energy vs (q_angle,r) index surface")
    _heat(fig_dir, Z_S, [0, 1, 0, 1], "Settling vs (q_angle,r) index surface")

    # （可选）热图 10：综合 tradeoff 指标，保证热图数量富余
    trade = Z_I / (Z_E + 1e-6)
    _heat(fig_dir, trade, [0, 1, 0, 1], "ISE/Energy tradeoff surface")

# ---------- 5. 时间响应 / 多曲线 ----------
def _generate_lineplots(ctx, quat, fig_dir: Path):
    # 四元数小角度
    xs_q_lqr = quat["xs_q_lqr"]
    xs_q_hinf = quat["xs_q_hinf"]
    _line_multi(fig_dir, xs_q_lqr[:1200, :3], "Quaternion small-angle (LQR)")
    _line_multi(fig_dir, xs_q_hinf[:1200, :3], "Quaternion small-angle (H∞)")
    # 角速度
    _line_multi(fig_dir, xs_q_lqr[:1200, 3:], "Quaternion angular velocity (LQR)")
    _line_multi(fig_dir, xs_q_hinf[:1200, 3:], "Quaternion angular velocity (H∞)")

    # 线性模型：不同控制器的角度范数响应
    A0 = ctx["A0"]
    B0 = ctx["B0"]
    Bw0 = ctx["Bw0"]
    K_lqr = ctx["K_lqr"]
    K_hinf = ctx["K_hinf"]
    policy_mpc = ctx["policy_mpc"]

    def angle_norm_response(K_or_policy, mode, title):
        if mode == "statefb":
            xs, us = simulate_linear(
                A0,
                B0,
                Bw0,
                K_or_policy,
                T=2.0,
                dt=0.01,
                disturb_scale=0.10,
                policy_type="statefb",
                u_limits=[-0.6, 0.6],
            )
        else:
            xs, us = simulate_linear(
                A0,
                B0,
                Bw0,
                K_or_policy,
                T=2.0,
                dt=0.01,
                disturb_scale=0.10,
                policy_type="mpc",
                u_limits=[-0.6, 0.6],
            )
        ang = np.linalg.norm(xs[:, :3], axis=1).reshape(-1, 1)
        _line_multi(fig_dir, np.repeat(ang, 3, axis=1), title)

    angle_norm_response(K_lqr, "statefb", "Angle norm response (LQR)")
    angle_norm_response(K_hinf, "statefb", "Angle norm response (H∞)")
    angle_norm_response(policy_mpc, "mpc", "Angle norm response (tMPC)")


# ---------- 6. 极点云 ----------
def _generate_poleplots(ctx, fig_dir: Path):
    K_lqr = ctx["K_lqr"]
    K_hinf = ctx["K_hinf"]
    K_sched = ctx["K_sched"]
    J_centers = ctx["J_centers"]

    def choose_K(Jd):
        d = [abs(Jd[0] - Jc[0]) for Jc, _ in K_sched]
        return K_sched[int(np.argmin(d))][1]

    def pole_cloud_for_K(K, title, seed=3):
        xs = []
        ys = []
        rng = np.random.default_rng(seed)
        for _ in range(80):
            Jd = np.array(
                [
                    rng.uniform(0.7, 1.3),
                    rng.uniform(0.8, 1.4),
                    rng.uniform(0.6, 1.2),
                ]
            )
            A, B, Bw, C, Du, Dw = satellite_linear_model(Jd, np.zeros((6, 6)))
            lam = np.linalg.eigvals(closed_loop(A, B, K))
            xs.extend(lam.real)
            ys.extend(lam.imag)
        _pole_cloud(fig_dir, xs, ys, title)

    for seed in [3, 5, 7]:
        pole_cloud_for_K(K_lqr, f"Pole cloud (LQR, seed={seed})", seed=seed)
    for seed in [11, 13, 17]:
        pole_cloud_for_K(K_hinf, f"Pole cloud (H∞, seed={seed})", seed=seed)

    def pole_cloud_sched(seed=21):
        xs = []
        ys = []
        rng = np.random.default_rng(seed)
        for _ in range(80):
            Jd = np.array(
                [
                    rng.uniform(0.7, 1.3),
                    rng.uniform(0.8, 1.4),
                    rng.uniform(0.6, 1.2),
                ]
            )
            Kc = choose_K(Jd)
            A, B, Bw, C, Du, Dw = satellite_linear_model(Jd, np.zeros((6, 6)))
            lam = np.linalg.eigvals(closed_loop(A, B, Kc))
            xs.extend(lam.real)
            ys.extend(lam.imag)
        _pole_cloud(fig_dir, xs, ys, f"Pole cloud (Sched, seed={seed})")

    for seed in [21, 23, 29]:
        pole_cloud_sched(seed=seed)


# ---------- 总入口：生成所有单图 ----------
def generate_all_figures(ctx: dict, dfs: dict, quat: dict, fig_dir: Path):
    _generate_boxplots(ctx, dfs, fig_dir)
    _generate_histograms(ctx, dfs, fig_dir)
    _generate_scatters(ctx, dfs, fig_dir)
    _generate_heatmaps(ctx, fig_dir)
    _generate_lineplots(ctx, quat, fig_dir)
    _generate_poleplots(ctx, fig_dir)


# ---------- 3×3 拼图：每类独立一张 ----------
def build_mosaics_by_type(fig_dir: Path):
    type_tags = ["box", "hist", "scat", "heat", "line", "pole"]

    for tag in type_tags:
        imgs = sorted(fig_dir.glob(f"{tag}_*.png"))
        if len(imgs) < 9:
            print(f"[WARN] type {tag} only has {len(imgs)} figs, need >=9.")
            continue
        sel = imgs[:9]
        out_path = fig_dir / f"mosaic_{tag}_3x3.png"
        stitch_3x3(sel, out_path)
