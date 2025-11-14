import numpy as np
def satellite_linear_model(J_diag=np.array([0.8,1.0,1.2]), deltaA=None):
    Z3 = np.zeros((3,3))
    A_top = np.hstack([Z3, np.eye(3)])
    Jinv = np.diag(1.0/J_diag)
    A_bottom = np.hstack([Z3, Z3])
    A = np.vstack([A_top, A_bottom])
    B = np.vstack([Z3, Jinv])
    if deltaA is not None: A = A + deltaA
    Bw = np.vstack([Z3, Jinv])
    Cx = np.eye(6)
    Du = np.zeros((6,3)); Dw = np.zeros((6,3))
    return A,B,Bw,Cx,Du,Dw

def draw_uncertainty(num=60, seed=2025):
    rng = np.random.default_rng(seed)
    Jx = rng.uniform(0.7,1.3,size=num)
    Jy = rng.uniform(0.8,1.4,size=num)
    Jz = rng.uniform(0.6,1.2,size=num)
    dAs = []
    for _ in range(num):
        d = np.zeros((6,6))
        d[0,4] = rng.normal(0,0.03)
        d[1,5] = rng.normal(0,0.03)
        d[2,3] = rng.normal(0,0.03)
        dAs.append(d)
    return np.stack([Jx,Jy,Jz],axis=1), dAs
