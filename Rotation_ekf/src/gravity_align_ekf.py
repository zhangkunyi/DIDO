# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180

def skew(x):
    if len(x) != 3:
        raise ValueError("Input must be an array with three elements")
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0.0]])

def sind(x):
    if isinstance(x, list):
        x = np.asarray(x)
    return np.sin(x*DEG2RAD)

def cosd(x):
    if isinstance(x, list):
        x = np.asarray(x)
    return np.cos(x*DEG2RAD)

def chiaverini(dcm: np.ndarray) -> np.ndarray:
    n = 0.5*np.sqrt(dcm.trace() + 1.0)
    e = np.array([0.5*np.sign(dcm[2, 1]-dcm[1, 2])*np.sqrt(dcm[0, 0]-dcm[1, 1]-dcm[2, 2]+1.0),
                  0.5*np.sign(dcm[0, 2]-dcm[2, 0])*np.sqrt(dcm[1, 1]-dcm[2, 2]-dcm[0, 0]+1.0),
                  0.5*np.sign(dcm[1, 0]-dcm[0, 1])*np.sqrt(dcm[2, 2]-dcm[0, 0]-dcm[1, 1]+1.0)])
    return np.array([n, *e])

def ecompass(a: np.ndarray, m: np.ndarray, frame: str = 'ENU', representation: str = 'rotmat') -> np.ndarray:
    if frame.upper() not in ['ENU', 'NED']:
        raise ValueError("Wrong local tangent plane coordinate frame. Try 'ENU' or 'NED'")
    if representation.lower() not in ['rotmat', 'quaternion', 'rpy', 'axisangle']:
        raise ValueError("Wrong representation type. Try 'rotmat', 'quaternion', 'rpy', or 'axisangle'")
    a = np.copy(a)
    m = np.copy(m)
    if a.shape[-1] != 3 or m.shape[-1] != 3:
        raise ValueError("Input vectors must have exactly 3 elements.")
    m /= np.linalg.norm(m)
    Rz = a/np.linalg.norm(a)
    if frame.upper() == 'NED':
        Ry = np.cross(Rz, m)
        Rx = np.cross(Ry, Rz)
    else:
        Rx = np.cross(m, Rz)
        Ry = np.cross(Rz, Rx)
    Rx /= np.linalg.norm(Rx)
    Ry /= np.linalg.norm(Ry)
    R = np.c_[Rx, Ry, Rz].T
    if representation.lower() == 'quaternion':
        return chiaverini(R)
    if representation.lower() == 'rpy':
        phi = np.arctan2(R[1, 2], R[2, 2])    # Roll Angle
        theta = -np.arcsin(R[0, 2])           # Pitch Angle
        psi = np.arctan2(R[0, 1], R[0, 0])    # Yaw Angle
        return np.array([phi, theta, psi])
    if representation.lower() == 'axisangle':
        angle = np.arccos((R.trace()-1)/2)
        axis = np.zeros(3)
        if angle!=0:
            S = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
            axis = S/(2*np.sin(angle))
        return (axis, angle)
    return R

def q2R(q: np.ndarray) -> np.ndarray:
    if q is None:
        return np.identity(3)
    if q.shape[-1]!= 4:
        raise ValueError("Quaternion Array must be of the form (4,) or (N, 4)")
    if q.ndim>1:
        q /= np.linalg.norm(q, axis=1)[:, None]     # Normalize all quaternions
        R = np.zeros((q.shape[0], 3, 3))
        R[:, 0, 0] = 1.0 - 2.0*(q[:, 2]**2 + q[:, 3]**2)
        R[:, 1, 0] = 2.0*(q[:, 1]*q[:, 2]+q[:, 0]*q[:, 3])
        R[:, 2, 0] = 2.0*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])
        R[:, 0, 1] = 2.0*(q[:, 1]*q[:, 2]-q[:, 0]*q[:, 3])
        R[:, 1, 1] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 3]**2)
        R[:, 2, 1] = 2.0*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3])
        R[:, 0, 2] = 2.0*(q[:, 1]*q[:, 3]+q[:, 0]*q[:, 2])
        R[:, 1, 2] = 2.0*(q[:, 2]*q[:, 3]-q[:, 0]*q[:, 1])
        R[:, 2, 2] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 2]**2)
        return R
    q /= np.linalg.norm(q)
    return np.array([
        [1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
        [2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[0]*q[1])],
        [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2)]])


def acc2q(a: np.ndarray, return_euler: bool = False) -> np.ndarray:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    ex, ey, ez = 0.0, 0.0, 0.0
    if np.linalg.norm(a)>0 and len(a)==3:
        ax, ay, az = a
        # Normalize accerometer measurements
        a_norm = np.linalg.norm(a)
        ax /= a_norm
        ay /= a_norm
        az /= a_norm
        # Euler Angles from Gravity vector
        ex = np.arctan2(ay, az)
        ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        ez = 0.0
        if return_euler:
            return np.array([ex, ey, ez])*RAD2DEG
        # Euler to Quaternion
        cx2 = np.cos(ex/2.0)
        sx2 = np.sin(ex/2.0)
        cy2 = np.cos(ey/2.0)
        sy2 = np.sin(ey/2.0)
        q = np.array([cx2*cy2, sx2*cy2, cx2*sy2, -sx2*sy2])
        q /= np.linalg.norm(q)
    return q


class gravity_align_EKF:

    def __init__(self,
                 gyr: np.ndarray = None,
                 acc: np.ndarray = None,
                 frequency: float = 100.0,
                 frame: str = 'NED',
                 acc_cov: np.ndarray = None,
                 update_rate: float = 100.0,
                 Init_state_cov: float = 1.0,
                 dt_seq: np.ndarray = None,
                 **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.frequency = frequency
        self.frame = frame  # Local tangent plane coordinate frame
        self.q0 = kwargs.get('q0')
        self.P = np.diag(np.repeat(Init_state_cov, 4))
        self.a_ref = np.array([0.0, 0.0, -1.0]) if frame.upper() == 'NED' else np.array([0.0, 0.0, 1.0])
        self.R = self._set_measurement_noise_covariance(**kwargs)
        self.acc_cov = acc_cov
        self.update_rate = update_rate
        self.update_interval = int(self.frequency / self.update_rate)
        self.dt_seq = dt_seq

        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all(self.frame)

    def _set_measurement_noise_covariance(self, **kw) -> np.ndarray:
        self.noises = np.array(kw.get('noises', [0.3 ** 2]))
        self.g_noise = self.noises


    def _compute_all(self, frame: str) -> np.ndarray:

        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        Q[0] = self.q0

        ###### Compute attitude with IMU architecture ######
        if self.q0 is None:
            Q[0] = acc2q(self.acc[0])
        Q[0] /= np.linalg.norm(Q[0])

        # update and correction are not at same time
        update_index = list(range(0, num_samples, self.update_interval))
        update_i = 1
        # Q_compensation
        Q_compensation = np.zeros((len(update_index), 4))
        Q_compensation[0] = [1, 0, 0, 0]

        # EKF Loop over all data
        for t in tqdm(range(1, num_samples)):
            Q[t] = self.Prediction(t, self.dt_seq[t - 1], Q[t - 1], self.gyr[t - 1],self.acc[t])
            if t == update_index[update_i]:
                Q[t] = self.Correction( Q[t])
                if update_index[update_i] < update_index[-1]:
                    update_i += 1

        return Q

    def Omega(self, x: np.ndarray) -> np.ndarray:

        return np.array([
            [0.0, -x[0], -x[1], -x[2]],
            [x[0], 0.0, x[2], -x[1]],
            [x[1], -x[2], 0.0, x[0]],
            [x[2], x[1], -x[0], 0.0]])

    def f(self, q: np.ndarray, omega: np.ndarray, dt_seq: np.ndarray) -> np.ndarray:

        Omega_t = self.Omega(omega)
        return (np.identity(4) + 0.5 * dt_seq * Omega_t) @ q

    def dfdq(self, omega: np.ndarray, dt_seq: np.ndarray) -> np.ndarray:

        x = 0.5 * dt_seq * omega
        return np.identity(4) + self.Omega(x)

    def h(self, q: np.ndarray) -> np.ndarray:
        C = q2R(q).T
        if len(self.z) < 4:
            return C @ self.a_ref
        return np.r_[C @ self.a_ref, C @ self.m_ref]

    def dhdq(self, q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q
        v = np.r_[self.a_ref]
        H = np.array([[-qy * v[2] + qz * v[1], qy * v[1] + qz * v[2], -qw * v[2] + qx * v[1] - 2.0 * qy * v[0],
                       qw * v[1] + qx * v[2] - 2.0 * qz * v[0]],
                      [qx * v[2] - qz * v[0], qw * v[2] - 2.0 * qx * v[1] + qy * v[0], qx * v[0] + qz * v[2],
                       -qw * v[0] + qy * v[2] - 2.0 * qz * v[1]],
                      [-qx * v[1] + qy * v[0], -qw * v[1] - 2.0 * qx * v[2] + qz * v[0],
                       qw * v[0] - 2.0 * qy * v[2] + qz * v[1], qx * v[0] + qy * v[1]]])
        if len(self.z) == 6:
            H_2 = np.array([[-qy * v[5] + qz * v[4], qy * v[4] + qz * v[5], -qw * v[5] + qx * v[4] - 2.0 * qy * v[3],
                             qw * v[4] + qx * v[5] - 2.0 * qz * v[3]],
                            [qx * v[5] - qz * v[3], qw * v[5] - 2.0 * qx * v[4] + qy * v[3], qx * v[3] + qz * v[5],
                             -qw * v[3] + qy * v[5] - 2.0 * qz * v[4]],
                            [-qx * v[4] + qy * v[3], -qw * v[4] - 2.0 * qx * v[5] + qz * v[3],
                             qw * v[3] - 2.0 * qy * v[5] + qz * v[4], qx * v[3] + qy * v[4]]])
            H = np.vstack((H, H_2))
        return 2.0 * H

    def Prediction(self, t: int, dt_seq: np.ndarray, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray,
                   mag: np.ndarray = None) -> np.ndarray:

        if not np.isclose(np.linalg.norm(q), 1.0):
            raise ValueError("A-priori quaternion must have a norm equal to 1.")
        # Current Measurements
        g = np.copy(gyr)  # Gyroscope data as control vector
        a = np.copy(acc)
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return q
        a /= a_norm
        self.z = np.copy(a)

        self.R = np.diag(np.repeat(self.acc_cov[t], 3))  # change cov of acc

        # ----- Prediction -----
        q_t = self.f(q, g, dt_seq)  # Predicted State
        F = self.dfdq(g, dt_seq)  # Linearized Fundamental Matrix
        W = 0.5 * dt_seq * np.r_[[-q[1:]], q[0] * np.identity(3) + skew(q[1:])]
        Q_t = 0.5 * dt_seq * self.g_noise * W @ W.T  # Process Noise Covariance
        P_t = F @ self.P @ F.T + Q_t  # Predicted Covariance Matrix
        self.P = P_t  # Predicted Covariance Matrix pass on

        self.q = q_t
        self.q /= np.linalg.norm(self.q)
        return self.q

    def Correction(self, q_t: np.ndarray = None) -> np.ndarray:
        P_t = self.P
        # ----- Correction -----
        y = self.h(q_t)  # Expected Measurement function
        v = self.z - y  # Innovation (Measurement Residual)
        H = self.dhdq(q_t)  # Linearized Measurement Matrix
        S = H @ P_t @ H.T + self.R  # Measurement Prediction Covariance

        K = P_t @ H.T @ np.linalg.inv(S)  # Kalman Gain

        self.P = (np.identity(4) - K @ H) @ P_t
        q_correct = q_t + K @ v  # Corrected state

        self.q = q_correct  # Corrected state
        self.q /= np.linalg.norm(self.q)

        return self.q