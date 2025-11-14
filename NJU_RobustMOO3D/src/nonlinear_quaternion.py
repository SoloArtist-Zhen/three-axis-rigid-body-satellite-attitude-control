import numpy as np
def euler_to_quat(angles):
    phi, theta, psi = angles
    c1,s1 = np.cos(psi/2), np.sin(psi/2)
    c2,s2 = np.cos(theta/2), np.sin(theta/2)
    c3,s3 = np.cos(phi/2), np.sin(phi/2)
    qw = c1*c2*c3 + s1*s2*s3
    qx = c1*c2*s3 - s1*s2*c3
    qy = c1*s2*c3 + s1*c2*s3
    qz = s1*c2*c3 - c1*s2*s3
    q = np.array([qw,qx,qy,qz]); return q/np.linalg.norm(q)
def quat_to_small_angles(q): return 2.0*q[1:]
def simulate_quaternion(J_diag, K, T=3.0, dt=0.002, torque_sat=0.5, noise_std=0.002, anti_windup=True):
    J = np.diag(J_diag); steps=int(T/dt)
    q = euler_to_quat(np.array([0.18,-0.12,0.15])); w = np.zeros(3)
    xs=np.zeros((steps,6)); us=np.zeros((steps,3)); aw=np.zeros(3)
    for k in range(steps):
        ang = quat_to_small_angles(q) + np.random.normal(0, noise_std, 3)
        x = np.hstack([ang, w + np.random.normal(0, noise_std, 3)])
        u = -K @ x + (aw if anti_windup else 0.0)
        u = np.clip(u, -torque_sat, torque_sat)
        if anti_windup: aw += 0.05*(u - (-K@x))
        jw = J@w
        wdot = np.linalg.solve(J, (u - np.cross(w, jw)))
        Omega = np.array([[0,-w[0],-w[1],-w[2]],[w[0],0,w[2],-w[1]],[w[1],-w[2],0,w[0]],[w[2],w[1],-w[0],0]])
        qdot = 0.5*Omega @ q
        w = w + dt*wdot; q = (q + dt*qdot); q = q/np.linalg.norm(q)
        xs[k,:3]=quat_to_small_angles(q); xs[k,3:]=w; us[k]=u
    return xs,us
