from pathlib import Path
import numpy as np
import pandas as pd

from .models import satellite_linear_model, draw_uncertainty
from .controllers import (
    lqr,
    hinf_state_feedback_surrogate,
    finite_horizon_lqr,
    simulate_linear,
    ise,
    energy,
    settling,
    closed_loop,
)
from .metrics import hinf_like, spectral_abscissa
from .nonlinear_quaternion import simulate_quaternion


def build_context(data_dir: Path):
    """
    建立整个工程需要的“上下文”：
    - 名义模型 A0,B0,Bw0...
    - LQR / H∞ 控制器
    - tube-MPC 名义控制器
    - 不确定性样本 + 多模型调度控制器
    """
    A0, B0, Bw0, C0, Du0, Dw0 = satellite_linear_model()
    Q = np.diag([10, 10, 10, 1, 1, 1])
    R = np.eye(3) * 0.5

    # 基准 LQR / 近似 H∞
    K_lqr, _ = lqr(A0, B0, Q, R)
    K_hinf, _ = hinf_state_feedback_surrogate(A0, B0, Bw0, Q, R, gamma=1.35)

    # tube-MPC 名义控制器（有限时域 LQR）
    Ks = finite_horizon_lqr(A0, B0, Q, R, N=8)
    policy_mpc = {"Ks": Ks, "Kt": K_lqr}

    # 不确定性 & 线性模型扰动
    J_pert, dA_list = draw_uncertainty(num=72, seed=2025)
    pd.DataFrame(J_pert, columns=["Jx", "Jy", "Jz"]).to_csv(
        data_dir / "uncertainty_samples.csv", index=False
    )

    # 多模型调度：简单按 Jx 分成 3 簇
    qs = np.quantile(J_pert[:, 0], [0.0, 1 / 3, 2 / 3, 1.0])
    K_sched = []
    J_centers = []

    for i in range(3):
        mask = (J_pert[:, 0] >= qs[i]) & (J_pert[:, 0] < qs[i + 1])
        if np.any(mask):
            Jc = np.median(J_pert[mask], axis=0)
        else:
            Jc = np.array([0.9, 1.1, 1.0])

        A, B, Bw, C, Du, Dw = satellite_linear_model(Jc, np.zeros((6, 6)))

        if i % 2 == 0:
            Kc, _ = lqr(A, B, Q, R)
        else:
            Kc, _ = hinf_state_feedback_surrogate(A, B, Bw, Q, R, gamma=1.3)

        K_sched.append((Jc, Kc))
        J_centers.append(Jc)

    ctx = {
        "A0": A0,
        "B0": B0,
        "Bw0": Bw0,
        "C0": C0,
        "Du0": Du0,
        "Dw0": Dw0,
        "Q": Q,
        "R": R,
        "K_lqr": K_lqr,
        "K_hinf": K_hinf,
        "Ks": Ks,
        "policy_mpc": policy_mpc,
        "J_pert": J_pert,
        "dA_list": dA_list,
        "K_sched": K_sched,
        "J_centers": J_centers,
    }
    return ctx


def run_all_mc(ctx: dict, res_dir: Path):
    """
    对四种控制器做 Monte Carlo：
    - LQR
    - H∞
    - tube-MPC（简化）
    - Scheduling（多模型调度）
    返回 dict: {"lqr": df_lqr, "hinf": df_hinf, "mpc": df_mpc, "sched": df_sched}
    """
    J_pert = ctx["J_pert"]
    dA_list = ctx["dA_list"]
    K_lqr = ctx["K_lqr"]
    K_hinf = ctx["K_hinf"]

    def choose_K(Jd):
        J_centers = ctx["J_centers"]
        K_sched = ctx["K_sched"]
        d = [abs(Jd[0] - Jc[0]) for Jc, _ in K_sched]
        return K_sched[int(np.argmin(d))][1]

    def mc(mode, T=2.2, dt=0.01, nsamp=28):
        rows = []
        idxs = np.linspace(0, len(J_pert) - 1, nsamp, dtype=int)

        for idx in idxs:
            Jd = J_pert[idx]
            dA = dA_list[idx]
            A, B, Bw, C, Du, Dw = satellite_linear_model(Jd, dA)

            if mode == "lqr":
                Kuse = K_lqr
                xs, us = simulate_linear(
                    A,
                    B,
                    Bw,
                    Kuse,
                    T=T,
                    dt=dt,
                    disturb_scale=0.12,
                    policy_type="statefb",
                    u_limits=[-0.6, 0.6],
                    rng=np.random.default_rng(idx),
                )
                Acl = closed_loop(A, B, Kuse)
            elif mode == "hinf":
                Kuse = K_hinf
                xs, us = simulate_linear(
                    A,
                    B,
                    Bw,
                    Kuse,
                    T=T,
                    dt=dt,
                    disturb_scale=0.12,
                    policy_type="statefb",
                    u_limits=[-0.6, 0.6],
                    rng=np.random.default_rng(idx),
                )
                Acl = closed_loop(A, B, Kuse)
            elif mode == "mpc":
                xs, us = simulate_linear(
                    A,
                    B,
                    Bw,
                    ctx["policy_mpc"],
                    T=T,
                    dt=dt,
                    disturb_scale=0.10,  # 稍微小一点，减少溢出
                    policy_type="mpc",
                    u_limits=[-0.6, 0.6],
                    rng=np.random.default_rng(idx),
                )
                Acl = closed_loop(A, B, K_lqr)  # 用 LQR 闭环做 freq 代理
            elif mode == "sched":
                Kc = choose_K(Jd)
                xs, us = simulate_linear(
                    A,
                    B,
                    Bw,
                    Kc,
                    T=T,
                    dt=dt,
                    disturb_scale=0.12,
                    policy_type="statefb",
                    u_limits=[-0.6, 0.6],
                    rng=np.random.default_rng(idx),
                )
                Acl = closed_loop(A, B, Kc)
            else:
                raise ValueError("Unknown mode: " + str(mode))

            I = ise(xs, dt)
            E = energy(us, dt)
            ST = settling(xs, dt)
            try:
                H = hinf_like(
                    Acl,
                    Bw,
                    C=np.eye(6),
                    D=np.zeros((6, 3)),
                    nfreq=60,
                )
            except Exception:
                H = np.nan
            SA = spectral_abscissa(Acl)

            rows.append(
                {
                    "ISE": I,
                    "Eff": E,  # 后面画图统一用 'Eff'
                    "Sett": ST,
                    "Hinf_like": H,
                    "SpecAbs": SA,
                }
            )

        df = pd.DataFrame(rows)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df.to_csv(res_dir / f"mc_{mode}.csv", index=False)
        return df

    df_lqr = mc("lqr")
    df_hinf = mc("hinf")
    df_mpc = mc("mpc")
    df_sched = mc("sched")

    return {
        "lqr": df_lqr,
        "hinf": df_hinf,
        "mpc": df_mpc,
        "sched": df_sched,
    }


def run_quaternion(ctx: dict):
    """非线性四元数仿真：LQR vs H∞"""
    K_lqr = ctx["K_lqr"]
    K_hinf = ctx["K_hinf"]

    xs_q_lqr, us_q_lqr = simulate_quaternion(
        [0.8, 1.0, 1.2],
        K_lqr,
        T=2.0,
        dt=0.002,
        torque_sat=0.5,
        noise_std=0.002,
    )
    xs_q_hinf, us_q_hinf = simulate_quaternion(
        [0.8, 1.0, 1.2],
        K_hinf,
        T=2.0,
        dt=0.002,
        torque_sat=0.5,
        noise_std=0.002,
    )

    return {
        "xs_q_lqr": xs_q_lqr,
        "us_q_lqr": us_q_lqr,
        "xs_q_hinf": xs_q_hinf,
        "us_q_hinf": us_q_hinf,
    }
