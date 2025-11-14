# robust/hinf_lmi_stub.py

import numpy as np
import cvxpy as cp


def hinf_state_feedback_lmi(A, B, Bw, gamma=1.0, eps=1e-6, solver=None, verbose=False):
    """
    LMI-based H∞ state-feedback synthesis for:
        xdot = A x + B u + Bw w,   u = -K x
    s.t. || T_{w->x} ||_∞ < gamma.

    Parameters
    ----------
    A, B, Bw : np.ndarray
        State, control and disturbance matrices.
    gamma : float
        Desired H∞ bound.
    eps : float
        Small positive constant to enforce P >> eps I.
    solver : str or None
        CVXPY solver name (e.g., "MOSEK", "OSQP", "SCS"). If None, CVXPY default.
    verbose : bool
        Print solver information.

    Returns
    -------
    K : np.ndarray
        State-feedback gain matrix.
    info : dict
        Extra info, including P, achieved gamma, and solver status.
    """
    n = A.shape[0]
    m = B.shape[1]
    nw = Bw.shape[1]

    # Decision variables
    P = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))

    # LMI:
    # [ A P + P A' + B Y + Y' B'   P Bw ]
    # [       Bw' P               -gamma^2 I ]  << 0
    AP = A @ P
    BY = B @ Y

    M11 = AP + AP.T + BY + BY.T
    M12 = P @ Bw
    M21 = M12.T
    M22 = - (gamma ** 2) * np.eye(nw)

    M = cp.bmat([[M11, M12],
                 [M21, M22]])

    constraints = [
        P >> eps * np.eye(n),
        M << 0
    ]

    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"H∞ LMI problem not solved to optimality: {prob.status}")

    P_val = P.value
    Y_val = Y.value
    K = Y_val @ np.linalg.inv(P_val)

    info = {
        "P": P_val,
        "status": prob.status,
        "gamma": gamma,
        "obj": prob.value,
    }
    return K, info
