import numpy as np
def hinf_like(A,Bw,C=None,D=None,wmin=1e-2,wmax=1e2,nfreq=160):
    n=A.shape[0]
    if C is None: C=np.eye(n)
    if D is None: D=np.zeros((C.shape[0], Bw.shape[1]))
    ws=np.logspace(np.log10(wmin), np.log10(wmax), nfreq)
    I=np.eye(n); peak=0.0
    for w in ws:
        G = C @ np.linalg.solve(1j*w*I - A, Bw) + D
        s = np.linalg.svd(G, compute_uv=False); peak=max(peak, float(np.max(s)))
    return peak
def spectral_abscissa(A): return float(np.max(np.linalg.eigvals(A).real))
