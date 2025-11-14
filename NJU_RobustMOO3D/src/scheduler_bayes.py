import numpy as np

class RLSJEstimator:
    """简单RLS演示：用粗略加速度响应来拟合惯量J的等效变化（示意用途）"""
    def __init__(self, lam=0.98, delta=100.0):
        self.P = np.eye(3)*delta
        self.theta = np.array([0.9,1.1,1.0])  # 初始Jx,Jy,Jz估计
        self.lam = lam
    def update(self, phi, y):
        # y ≈ phi^T * (1/J) 之类的关系（演示）-> 拟合等效参数
        phi = np.asarray(phi)
        K = self.P @ phi / (self.lam + phi.T @ self.P @ phi)
        err = y - phi.T @ self.theta
        self.theta = self.theta + K * err
        self.P = (self.P - np.outer(K, phi.T) @ self.P)/self.lam
        return self.theta
