import numpy as np
from scipy.linalg import solve_continuous_are

def lqr(A,B,Q,R):
    X = solve_continuous_are(A,B,Q,R)
    K = np.linalg.solve(R, B.T @ X)
    return K, X

def closed_loop(A,B,K): return A - B @ K

# H∞（ARE近似，不需外部求解器；真·LMI接口见 robust/hinf_lmi_stub.py）
def hinf_state_feedback_surrogate(A,Bu,Bw,Q,R, gamma=1.5):
    Qeff = Q + (1.0/(gamma**2)) * (Bw @ Bw.T)
    X = solve_continuous_are(A,Bu,Qeff,R)
    K = np.linalg.solve(R, Bu.T @ X)
    return K, X

# Tube-MPC（简化）：有限时域 nominal LQR + 饱和/反饱和接口（QP/几率约束入口留在注释处）
def finite_horizon_lqr(A,B,Q,R,N):
    P = Q.copy(); Ks=[]
    for _ in range(N):
        X = solve_continuous_are(A,B,P,R)
        K = np.linalg.solve(R, B.T @ X); Ks.append(K)
        P = Q + (A - B@K).T @ X @ (A - B@K)
    Ks = Ks[::-1]; return Ks

def tube_mpc_step(A,B,x, Ks, Kt, u_limits=None):
    # TODO: 若安装了 OSQP/qp 求解器，可在此求解 QP 并添加几率约束/SAA
    u = -Ks[0] @ x
    if u_limits is not None: u = np.clip(u, u_limits[0], u_limits[1])
    return u

def simulate_linear(A,B,Bw,K_or_policy, T=3.0, dt=0.01, x0=None, disturb_scale=0.1, policy_type="statefb", u_limits=None, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    if x0 is None: x0 = np.array([0.15,-0.12,0.18,0,0,0])
    steps = int(T/dt); x = x0.copy()
    xs = np.zeros((steps,A.shape[0])); us = np.zeros((steps,B.shape[1]))
    for k in range(steps):
        w = disturb_scale*rng.normal(0,0.5,size=(Bw.shape[1],))
        if policy_type=="statefb":
            u = -K_or_policy @ x
        elif policy_type=="mpc":
            Ks = K_or_policy["Ks"]; Kt=K_or_policy["Kt"]
            u = tube_mpc_step(A,B,x, Ks, Kt, u_limits=u_limits)
        else:
            u = np.zeros(B.shape[1])
        if u_limits is not None: u = np.clip(u, u_limits[0], u_limits[1])
        x = x + dt*(A@x + B@u + Bw@w)
        xs[k]=x; us[k]=u
    return xs,us

def ise(xs,dt): return float(np.sum(xs**2)*dt)
def energy(us,dt): return float(np.sum(us**2)*dt)
def settling(xs,dt,tol=0.02):
    m = np.linalg.norm(xs[:,:3],axis=1)
    idx = np.where(m>tol)[0]
    return 0.0 if len(idx)==0 else float((idx[-1]+1)*dt)
