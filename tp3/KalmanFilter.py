import numpy as np
class KalmanFilter():
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.u_k = np.array([u_x, u_y])
        self.x_k = np.array([[0], [0], [0], [0]]) # x0, y0, vx0, vy0
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.B = np.array([
            [(dt**2)/2, 0],
            [0, (dt**2)/2],
            [dt, 0],
            [0, dt]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.array([
            [(dt**4)/4, 0, (dt**3)/2, 0],
            [0, (dt**4)/4, 0, (dt**3)/2],
            [(dt**3)/2, 0, dt**2, 0],
            [0, (dt**3)/2, 0, dt**2]
        ]) * std_acc**2
        self.R = np.array([
            [x_std_meas**2, 0],
            [0, y_std_meas**2]
        ])
        self.P = np.eye(self.A.shape[1])
    def predict(self):
        x_k = self.A @ self.x_k + self.B @ self.u_k
        P_k = self.A @ self.P @ self.A.T + self.Q
        self.x_k = x_k
        self.P = P_k
        return None
    def update(self, z_k):
        S_k = self.H @ self.P @ self.H.T + self.R
        K_k = self.P @ self.H.T @ np.linalg.inv(S_k)

        x_k = self.x_k + K_k @ (z_k - self.H @ self.x_k)
        p_k = (np.eye(self.H.shape[1]) - K_k @ self.H) @ self.P
        self.x_k = x_k
        self.P = p_k
        return None
