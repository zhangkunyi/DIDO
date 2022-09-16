import os

import torch
from network.model_factory import get_model
import numpy as np
from os import path as osp
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy import signal
from tqdm import tqdm
from scipy.linalg import block_diag
import argparse

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


STATE = 'p_v_k_drag_rot_trans'
gravity = np.array([0., 0., 9.8])

class Translation_EKF:
    def __init__(self,
                 p_init: np.ndarray = None,
                 v_init: np.ndarray = None,
                 tao_init: np.ndarray = None,
                 drag_init: np.ndarray = None,
                 rot_init: np.ndarray = None,
                 trans_init: np.ndarray = None,
                 state_cov: np.ndarray = None,
                 state_num: int = None,
                 ):
        self.p = p_init
        self.v = v_init
        self.ba = np.array([0, 0, 0])
        self.tao = tao_init
        self.drag = drag_init
        self.rot = rot_init
        self.trans = trans_init
        self.state_cov = state_cov

        self.state_num = state_num
        self.net_d_acc_all = []
        self.a_all = []

    def compute_all(self, dcm_imu, dt, rpm, R, w, alpha, obs_a=None, obs_v=None, obs_p=None, net_dyn_output=np.array([0,0,0]), Q = np.array([1e-2,1e-2,1e-2])):
        self.Q = Q
        self.dcm_rigid = np.matmul(dcm_imu,Rotation.from_quat(self.rot[[1, 2, 3, 0]]).as_matrix())
        p_pred, v_pred, ba_pred, tao_pred, drag_pred, rot_pred, trans_pred = self.process(self.dcm_rigid, dt, rpm, net_dyn_output)
        self.R = R
        if STATE == 'p_v_k_drag_rot_trans':
            state_pred = np.concatenate((p_pred, v_pred, tao_pred, drag_pred, rot_pred, trans_pred), axis=0)
            jacob_Fx = self.jacob_Fx_p_v_k_drag_q_t(self.dcm_rigid, rpm, dt, net_dyn_output)
            jacob_Fi = self.jacob_Fi_p_v_k_drag_q_t_to_a(self.dcm_rigid, dt)
            state_cov_pred = self.process_cov(jacob_Fx, jacob_Fi)
            if obs_a is not None and obs_v is None and obs_p is None:
                print('update_a')
                jacob_H = np.concatenate(
                    (np.zeros((3, 3)), self.jacob_h_acc_v(self.dcm_rigid, rpm, drag_pred, rot_pred),
                     self.jacob_h_acc_k(rpm, rot_pred),
                     self.jacob_h_acc_drag(self.dcm_rigid, rpm, v_pred, rot_pred),
                     self.jacob_h_acc_q(self.dcm_rigid, rpm, tao_pred, drag_pred, v_pred, rot_pred, net_dyn_output),
                     self.jacob_h_acc_t(w, alpha)), axis=1)
                K = self.kalman_gain(state_cov_pred, jacob_H)
                h = self.h_a(self.dcm_rigid, rpm, w, alpha, v_pred, tao_pred, drag_pred, rot_pred, trans_pred, net_dyn_output)
                obs = obs_a
                state_cur, self.state_cov = self.update(state_pred, state_cov_pred, K, obs, h, jacob_H)
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = state_cur[0:3], state_cur[3:6], state_cur[6:7], state_cur[7:10], state_cur[10:14], state_cur[14:17]
            elif obs_a is None and obs_v is not None and obs_p is None:
                print('update_v')
                jacob_H = np.concatenate((np.zeros((3, 3)), self.jacob_h_v_v(), np.zeros((3, 4 + 3)),
                                          self.jacob_h_v_t(w, dcm_imu, trans_pred)), axis=1)
                K = self.kalman_gain(state_cov_pred, jacob_H)
                h_v = self.h_v(dcm_imu, w, v_pred, trans_pred)
                h = h_v
                obs = obs_v
                state_cur, self.state_cov = self.update(state_pred, state_cov_pred, K, obs, h, jacob_H)
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = state_cur[0:3], state_cur[3:6], state_cur[6:7], state_cur[7:10], state_cur[10:14], state_cur[14:17]
            elif obs_a is None and obs_v is None and obs_p is not None:
                print('update_p')
                jacob_H = np.concatenate(
                    (self.jacob_h_p_p(), np.zeros((3, 7 + 3)), self.jacob_h_p_t(dcm_imu, w)), axis=1)
                K = self.kalman_gain(state_cov_pred, jacob_H)
                h_p = self.h_p(dcm_imu, p_pred, trans_pred)
                h = h_p
                obs = obs_p
                state_cur, self.state_cov = self.update(state_pred, state_cov_pred, K, obs, h, jacob_H)
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = state_cur[0:3], state_cur[3:6], state_cur[6:7], state_cur[7:10], state_cur[10:14], state_cur[14:17]
            elif obs_a is None and obs_v is not None and obs_p is not None:
                print('update_v_p')
                jacob_H_p = np.concatenate(
                    (self.jacob_h_p_p(), np.zeros((3, 7 + 3)), self.jacob_h_p_t(dcm_imu, w)), axis=1)
                jacob_H_v = np.concatenate((np.zeros((3, 3)), self.jacob_h_v_v(), np.zeros((3, 4 + 3)),
                                            self.jacob_h_v_t(w, dcm_imu, trans_pred)), axis=1)
                jacob_H = np.concatenate((jacob_H_p, jacob_H_v), axis=0)
                K = self.kalman_gain(state_cov_pred, jacob_H)
                h_p = self.h_p(dcm_imu, p_pred, trans_pred)
                h_v = self.h_v(dcm_imu, w, v_pred, trans_pred)
                h = np.concatenate((h_p, h_v), axis=0)
                obs = np.concatenate((obs_p, obs_v), axis=0)
                state_cur, self.state_cov = self.update(state_pred, state_cov_pred, K, obs, h, jacob_H)
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = state_cur[0:3], state_cur[3:6], state_cur[
                                                                                                          6:7], state_cur[
                                                                                                                7:10], state_cur[
                                                                                                                       10:14], state_cur[
                                                                                                                               14:17]
            elif obs_a is not None and obs_v is not None and obs_p is not None:
                print('update_p_v_a')
                jacob_H_p = np.concatenate(
                    (self.jacob_h_p_p(), np.zeros((3, 7 + 3)), self.jacob_h_p_t(dcm_imu, w)), axis=1)
                jacob_H_v = np.concatenate((np.zeros((3, 3)), self.jacob_h_v_v(), np.zeros((3, 4 + 3)),
                                            self.jacob_h_v_t(w, dcm_imu, trans_pred)), axis=1)
                jacob_H_a = np.concatenate(
                    (np.zeros((3, 3)), self.jacob_h_acc_v(self.dcm_rigid, rpm, drag_pred, rot_pred),
                     self.jacob_h_acc_k(rpm, rot_pred),
                     self.jacob_h_acc_drag(self.dcm_rigid, rpm, v_pred, rot_pred),
                     self.jacob_h_acc_q(self.dcm_rigid, rpm, tao_pred, drag_pred, v_pred, rot_pred, net_dyn_output),
                     self.jacob_h_acc_t(w, alpha)), axis=1)
                jacob_H = np.concatenate((jacob_H_p, jacob_H_v, jacob_H_a), axis=0)
                K = self.kalman_gain(state_cov_pred, jacob_H)
                h_p = self.h_p(dcm_imu, p_pred, trans_pred)
                h_v = self.h_v(dcm_imu, w, v_pred, trans_pred)
                h_a = self.h_a(self.dcm_rigid, rpm, w, alpha, v_pred, tao_pred, drag_pred, rot_pred, trans_pred, net_dyn_output)
                h = np.concatenate((h_p, h_v, h_a), axis=0)
                obs = np.concatenate((obs_p, obs_v, obs_a), axis=0)
                state_cur, self.state_cov = self.update(state_pred, state_cov_pred, K, obs, h, jacob_H)
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = state_cur[0:3], state_cur[3:6], state_cur[6:7], state_cur[7:10], state_cur[10:14], state_cur[14:17]
            elif obs_a is not None and obs_v is not None and obs_p is None:
                print('update_v_a')
                jacob_H_v = np.concatenate((np.zeros((3, 3)), self.jacob_h_v_v(), np.zeros((3, 4 + 3)),
                                            self.jacob_h_v_t(w, dcm_imu, trans_pred)), axis=1)
                jacob_H_a = np.concatenate(
                    (np.zeros((3, 3)), self.jacob_h_acc_v(self.dcm_rigid, rpm, drag_pred, rot_pred),
                     self.jacob_h_acc_k(rpm, rot_pred),
                     self.jacob_h_acc_drag(self.dcm_rigid, rpm, v_pred, rot_pred),
                     self.jacob_h_acc_q(self.dcm_rigid, rpm, tao_pred, drag_pred, v_pred, rot_pred, net_dyn_output),
                     self.jacob_h_acc_t(w, alpha)), axis=1)
                jacob_H = np.concatenate((jacob_H_v, jacob_H_a), axis=0)
                K = self.kalman_gain(state_cov_pred, jacob_H)
                h_v = self.h_v(dcm_imu, w, v_pred, trans_pred)
                h_a = self.h_a(self.dcm_rigid, rpm, w, alpha, v_pred, tao_pred, drag_pred, rot_pred, trans_pred, net_dyn_output)
                h = np.concatenate((h_v, h_a), axis=0)
                obs = np.concatenate((obs_v, obs_a), axis=0)
                state_cur, self.state_cov = self.update(state_pred, state_cov_pred, K, obs, h, jacob_H)
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = state_cur[0:3], state_cur[3:6], state_cur[6:7], state_cur[7:10], state_cur[10:14], state_cur[14:17]
            elif obs_a is not None and obs_v is None and obs_p is not None:
                print('update_p_a')
                jacob_H_p = np.concatenate(
                    (self.jacob_h_p_p(), np.zeros((3, 7 + 3)), self.jacob_h_p_t(dcm_imu, w)), axis=1)
                jacob_H_a = np.concatenate(
                    (np.zeros((3, 3)), self.jacob_h_acc_v(self.dcm_rigid, rpm, drag_pred, rot_pred),
                     self.jacob_h_acc_k(rpm, rot_pred),
                     self.jacob_h_acc_drag(self.dcm_rigid, rpm, v_pred, rot_pred),
                     self.jacob_h_acc_q(self.dcm_rigid, rpm, tao_pred, drag_pred, v_pred, rot_pred, net_dyn_output),
                     self.jacob_h_acc_t(w, alpha)), axis=1)
                jacob_H = np.concatenate((jacob_H_p, jacob_H_a), axis=0)
                K = self.kalman_gain(state_cov_pred, jacob_H)
                h_p = self.h_p(dcm_imu, p_pred, trans_pred)
                h_a = self.h_a(self.dcm_rigid, rpm, w, alpha, v_pred, tao_pred, drag_pred, rot_pred, trans_pred, net_dyn_output)
                h = np.concatenate((h_p, h_a), axis=0)
                obs = np.concatenate((obs_p, obs_a), axis=0)
                state_cur, self.state_cov = self.update(state_pred, state_cov_pred, K, obs, h, jacob_H)
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = state_cur[0:3], state_cur[3:6], state_cur[6:7], state_cur[7:10], state_cur[10:14], state_cur[14:17]
            else:
                self.p, self.v, self.tao, self.drag, self.rot, self.trans = p_pred, v_pred, tao_pred, drag_pred, rot_pred, trans_pred
                state_cur = np.concatenate((p_pred, v_pred, tao_pred, drag_pred, rot_pred, trans_pred), axis=0)
                self.state_cov = state_cov_pred

        return state_cur, self.state_cov

    def update(self, state_pred, state_cov_pred, K, obs, h, jacob_H):

        res = K @ (obs - h)
        state_cur_part_1 = state_pred[0:10] + res[0:10]
        q_res = np.array([1, 0.5 * res[10], 0.5 * res[11], 0.5 * res[12]])
        q_res = q_res / np.linalg.norm(q_res)
        state_cur_part_2 = qmul(state_pred[10:14].reshape(1, 4), q_res.reshape(1, 4))[0]
        state_cur_part_3 = state_pred[14:17] + res[13:16]
        state_cur = np.concatenate((state_cur_part_1, state_cur_part_2, state_cur_part_3), axis=0)
        state_cov_cur = (np.eye(state_cov_pred.shape[0]) - K @ jacob_H) @ state_cov_pred

        return state_cur, state_cov_cur

    def kalman_gain(self, state_cov_pred, H):
        k = state_cov_pred @ H.T @ np.linalg.inv(self.R + H @ state_cov_pred @ H.T)
        return k

    def process(self, dcm, dt, rpm, net_dyn_output):
        p = self.p + self.v * dt
        uss = np.sum(rpm ** 2)
        us = np.sum(rpm)
        v = self.v + (dcm @ (self.tao * uss * np.array([0., 0., 1.]) - us * np.diag(
            self.drag) @ dcm.T @ self.v + net_dyn_output) - gravity) * dt
        ba = self.ba
        k = self.tao
        drag = self.drag
        rot = self.rot
        trans = self.trans

        self.net_d_acc_all.append(self.tao * uss * np.array([0., 0., 1.]) - us * np.diag(
            self.drag) @ dcm.T @ self.v + net_dyn_output)
        self.a_all.append(self.tao * uss * np.array([0., 0., 1.]) - us * np.diag(
            self.drag) @ dcm.T @ self.v)

        return p, v, ba, k, drag, rot, trans

    def process_cov(self, jacob_Fx, jacob_Fi):
        state_cov_pred = jacob_Fx @ self.state_cov @ jacob_Fx.T + jacob_Fi @ self.Q @ jacob_Fi.T
        return state_cov_pred

    def h_p(self, dcm_imu, p_pred, trans_pred):
        trans_gra = dcm_imu @ trans_pred
        return p_pred - trans_gra

    def h_v(self, dcm_imu, w, v_pred, trans_pred):
        gyr_gra = dcm_imu @ w
        trans_gra = dcm_imu @ trans_pred
        return v_pred - np.cross(gyr_gra, trans_gra)

    def h_a(self, dcm, rpm, w, alpha, v_pred, tao_pred, drag_pred, rot_pred, trans_pred, net_dyn_output):
        us = np.sum(rpm)
        uss = np.sum(rpm ** 2)
        return Rotation.from_quat(rot_pred[[1, 2, 3, 0]]).as_matrix() @ (
                    tao_pred * uss * np.array([0., 0., 1]) - us * np.diag(drag_pred) @ dcm.T @ v_pred + net_dyn_output) \
               - np.cross(w, np.cross(w, trans_pred)) - np.cross(alpha, trans_pred)

    def jacob_Fx_p_v_k_drag(self, dcm, rpm, dt):
        jacob_p = np.concatenate((np.zeros((3, 3)), self.jacob_f_p_v(), np.zeros((3, 4))), axis=1)
        jacob_v = np.concatenate(
            (np.zeros((3, 3)), self.jacob_f_v_v(dcm, rpm), self.jacob_f_v_k(dcm, rpm), self.jacob_f_v_drag(dcm, rpm)),
            axis=1)
        jacob_k = np.zeros((1, 10))
        jacob_drag = np.zeros((3, 10))
        return np.eye(10) + np.concatenate((jacob_p, jacob_v, jacob_k, jacob_drag), axis=0) * dt

    def jacob_Fx_p_v_k_drag_q_t(self, dcm, rpm, dt, net_dyn_output):
        jacob_p = np.concatenate((np.zeros((3, 3)), self.jacob_f_p_v(), np.zeros((3, 4 + 3 + 3))), axis=1)
        jacob_v = np.concatenate(
            (np.zeros((3, 3)), self.jacob_f_v_v(dcm, rpm), self.jacob_f_v_k(dcm, rpm), self.jacob_f_v_drag(dcm, rpm),
             self.jacob_f_v_q(dcm, rpm, net_dyn_output), np.zeros((3, 3))),
            axis=1)
        jacob_k = np.zeros((1, 10 + 3 + 3))
        jacob_drag = np.zeros((3, 10 + 3 + 3))
        jacob_q = np.zeros((3, 10 + 3 + 3))
        jacob_t = np.zeros((3, 10 + 3 + 3))
        return np.eye(10 + 3 + 3) + np.concatenate((jacob_p, jacob_v, jacob_k, jacob_drag, jacob_q, jacob_t),
                                                   axis=0) * dt

    def jacob_Fi_p_v_k_drag(self, dcm, rpm, dt):
        jacob_p = np.zeros((3, 4))
        jacob_v = self.jacob_f_v_rpm(dcm, rpm)
        jacob_k = np.zeros((1, 4))
        jacob_drag = np.zeros((3, 4))
        return np.concatenate((jacob_p, jacob_v, jacob_k, jacob_drag), axis=0) * dt

    def jacob_Fi_p_v_k_drag_to_a(self, dcm, dt):
        jacob_p = np.zeros((3, 3))
        jacob_v = dcm
        jacob_k = np.zeros((1, 3))
        jacob_drag = np.zeros((3, 3))
        return np.concatenate((jacob_p, jacob_v, jacob_k, jacob_drag), axis=0) * dt

    def jacob_Fi_p_v_k_drag_q_t(self, dcm, rpm, dt):
        jacob_p = np.zeros((3, 4))
        jacob_v = self.jacob_f_v_rpm(dcm, rpm)
        jacob_k = np.zeros((1, 4))
        jacob_drag = np.zeros((3, 4))
        jacob_q = np.zeros((3, 4))
        jacob_t = np.zeros((3, 4))
        return np.concatenate((jacob_p, jacob_v, jacob_k, jacob_drag, jacob_q, jacob_t), axis=0) * dt

    def jacob_Fi_p_v_k_drag_q_t_to_a(self, dcm, dt):
        jacob_p = np.zeros((3, 3))
        jacob_v = dcm
        jacob_k = np.zeros((1, 3))
        jacob_drag = np.zeros((3, 3))
        jacob_q = np.zeros((3, 3))
        jacob_t = np.zeros((3, 3))
        return np.concatenate((jacob_p, jacob_v, jacob_k, jacob_drag, jacob_q, jacob_t), axis=0) * dt

    def jacob_Fx_v(self, dcm, rpm, dt):
        jacob_v = self.jacob_f_v_v(dcm, rpm)
        return np.eye(3) + jacob_v * dt

    def jacob_Fi_v(self, dcm, rpm, dt):
        jacob_v = self.jacob_f_v_rpm(dcm, rpm)
        return jacob_v * dt

    def jacob_Fx_v_k_drag(self, dcm, rpm, dt):
        jacob_v = np.concatenate(
            (self.jacob_f_v_v(dcm, rpm), self.jacob_f_v_k(dcm, rpm), self.jacob_f_v_drag(dcm, rpm)), axis=1)
        jacob_k = np.zeros((1, 7))
        jacob_drag = np.zeros((3, 7))
        return np.eye(7) + np.concatenate((jacob_v, jacob_k, jacob_drag), axis=0) * dt

    def jacob_Fi_v_k_drag(self, dcm, rpm, dt):
        jacob_v = self.jacob_f_v_rpm(dcm, rpm)
        jacob_k = np.zeros((1, 4))
        jacob_drag = np.zeros((3, 4))
        return np.concatenate((jacob_v, jacob_k, jacob_drag), axis=0) * dt

    def jacob_f_p_v(self):
        # 3 x 3
        return np.eye(3)

    def jacob_f_v_v(self, dcm, rpm):
        # 3 x 3
        us = np.sum(rpm)
        return dcm @ (- us * np.diag(self.drag)) @ dcm.T

    def jacob_f_v_k(self, dcm, rpm):
        # 3 x 1
        uss = np.sum(rpm ** 2)
        return uss * dcm[:, 2].reshape(3, 1)

    def jacob_f_v_drag(self, dcm, rpm):
        # 3 x 3
        us = np.sum(rpm)
        return dcm * (- us * (dcm.T @ self.v)).reshape(1, 3).repeat(3, axis=0)

    def jacob_f_v_rpm(self, dcm, rpm):
        # 3 x 4
        temp1 = - dcm @ (np.diag(self.drag) @ dcm.T @ self.v).reshape(3, 1).repeat(4, axis=1)  # bug
        temp2 = 2 * self.tao * dcm[:, 2].reshape(3, 1) @ rpm.reshape(1, 4)
        return temp1 + temp2

    def jacob_f_v_q(self, dcm, rpm, net_dyn_output):
        uss = np.sum(rpm ** 2)
        us = np.sum(rpm)
        part1 = uss * self.tao * dcm @ skew(np.array([0., 0., 1]))
        part2 = us * dcm @ skew(np.diag(self.drag) @ dcm.T @ self.v)
        part3 = us * dcm @ np.diag(self.drag) @ skew(dcm.T @ self.v)      ############ 之前part3没加.T
        part4 = dcm @ skew(net_dyn_output)
        return - part1 + part2 - part3 - part4

    def jacob_h_p_p(self):
        return np.eye(3)

    def jacob_h_p_t(self, dcm_imu, trans_pred):
        return - dcm_imu

    def jacob_h_v_v(self):
        return np.eye(3)

    def jacob_h_v_t(self, w, dcm_imu, trans_pred):
        return - dcm_imu @ skew(w)

    def jacob_h_acc_v(self, dcm, rpm, drag_pred, rot_pred):
        us = np.sum(rpm)
        return - Rotation.from_quat(rot_pred[[1, 2, 3, 0]]).as_matrix() @ (us * np.diag(drag_pred) @ dcm.T)

    def jacob_h_acc_ba(self):
        return np.eye(3)

    def jacob_h_acc_k(self, rpm, rot_pred):
        uss = np.sum(rpm ** 2)
        return (Rotation.from_quat(rot_pred[[1, 2, 3, 0]]).as_matrix() @ np.array([0., 0., uss])).reshape(3, 1)

    def jacob_h_acc_drag(self, dcm, rpm, v_pred, rot_pred):
        us = np.sum(rpm)
        R1 = Rotation.from_quat(rot_pred[[1, 2, 3, 0]]).as_matrix()
        col_1 = (- us * R1[:, 0:1] @ dcm[:, 0:1].T @ v_pred).reshape(3, 1)
        col_2 = (- us * R1[:, 1:2] @ dcm[:, 1:2].T @ v_pred).reshape(3, 1)
        col_3 = (- us * R1[:, 2:3] @ dcm[:, 2:3].T @ v_pred).reshape(3, 1)
        return np.concatenate((col_1, col_2, col_3), axis=1)

    def jacob_h_acc_q(self, dcm, rpm, tao_pred, drag_pred, v_pred, rot_pred, net_dyn_output):
        uss = np.sum(rpm ** 2)
        us = np.sum(rpm)
        return - Rotation.from_quat(rot_pred[[1, 2, 3, 0]]).as_matrix() @ skew(
            tao_pred * uss * np.array([0., 0., 1]) - us * np.diag(drag_pred) @ dcm.T @ v_pred + net_dyn_output) \
               - us * Rotation.from_quat(rot_pred[[1, 2, 3, 0]]).as_matrix() @ np.diag(drag_pred) @ skew(dcm.T @ v_pred)

    def jacob_h_acc_t(self, w, alpha):
        return - skew(w) @ skew(w) - skew(alpha)

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def qmul(q, r):
    """
    fork form https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py#L36
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    if type(q) is not np.ndarray:
        terms = torch.bmm(r.contiguous().view(-1, 4, 1), q.contiguous().view(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape).to(q.device)

    else:
        terms = np.matmul(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))
        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return np.stack((w, x, y, z), axis=1).reshape(original_shape)

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

def create_dir(file_name, fold_names):
    for fold_name in fold_names:
        if not osp.isdir(file_name + fold_name):
            os.makedirs(file_name + fold_name)
            print('create '+ file_name + fold_name)
        else:
            print('fold has existed')

def translation_ekf(update_a,use_net_d,update_v,update_p,args):
    root_dir = args.root_dir
    test_path = args.test_list
    data_path_s = get_datalist(test_path)
    device = args.device
    net_config = {"in_dim": int((0.05 * args.imu_freq) // 4)}
    network_d = get_model("resnet", net_config, input_dim=int(10), output_dim=int(3)).to(device)
    checkpoint = torch.load(args.network_dyn_path,map_location="cuda")  # 载入预训练模型
    network_d.load_state_dict(checkpoint["model_state_dict"])
    network_d.eval()

    p_error = []
    p_rigid_error = []
    v_error = []
    v_rigid_error = []
    a_rigid_error = []
    v_rigid_body_error = []

    v_factor = args.v_factor
    p_factor = args.p_factor
    file_name = args.out_dir
    fold_names = ["/tao_d","/p_imu_world","/p_rigid_world","/v_imu_world","/v_rigid_world","/rot","/trans","/result"]
    create_dir(file_name, fold_names)

    for data_path in data_path_s:
        v_rigid_body = []
        dcm_rigid_all = []
        with h5py.File(root_dir + '/' + data_path + '/data.hdf5', "r") as f:
            ts = np.copy(f["ts"])  # timestamp
            gt_p = np.copy(f["gt_p"])  # position in world frame
            gt_v = np.copy(f["gt_v"])  # velocity in world frame
            gt_q = np.copy(f["gt_q"])  # quaternion of body frame in world frame
            meas_rpm = np.copy(f["meas_rpm"])
            gt_acc = np.copy(f["gt_acc"])
            gt_gyr = np.copy(f["gt_gyr"])
            gt_alpha = np.copy(f["gt_alpha"])
            dynamic_param = np.copy(f["dynamic_params"])

        start_step = 0

        dt = np.diff(ts)
        dt = np.append(dt[0], dt).reshape(-1, 1)

        optim_trans = dynamic_param[4:7]  # 在imu系下看，imu到rigid的平移
        optim_pitch = dynamic_param[7]
        optim_roll = dynamic_param[8]

        optim_rot_dcm = Rotation.from_euler("xyz", np.array([optim_roll, optim_pitch, 0]),degrees=True).as_matrix().T  # rigid2imu

        gt_R = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]]).as_matrix()
        gt_v_imu_body = np.einsum("tpi,tp->ti", gt_R, gt_v)
        gt_v_rigid_body = np.einsum('ip,tp->ti', optim_rot_dcm.T, gt_v_imu_body)
        spline_imu_a = np.einsum("tpi,tp->ti", gt_R, gt_acc + gravity)

        with open(osp.join(args.network_gyr_path + data_path + "_gyr.txt"),encoding='utf-8') as f:
            net_gyr_imu = np.loadtxt(f, delimiter=",")
        with open(osp.join(args.network_acc_path + data_path + "_acc.txt"),encoding='utf-8') as f:
            net_acc_imu = np.loadtxt(f, delimiter=",")
        with open(osp.join(args.network_q_path + data_path + "_ekf_q.txt"),encoding='utf-8') as f:
            net_q_imu = np.loadtxt(f, delimiter=",")
        with open(osp.join(args.network_v_p_path, data_path + "_vp.txt"),encoding='utf-8') as f:
            net_v_p_imu_world = np.loadtxt(f, delimiter=",")
        net_v_v_cov_t_imu_world = net_v_p_imu_world[:, 0]
        net_v_imu_world = net_v_p_imu_world[:, 1:4]
        net_p_imu_world = net_v_p_imu_world[:, 4:7]
        with open(osp.join(args.network_v_p_cov_path, data_path + "_vp_cov.txt"),encoding='utf-8') as f:
            net_v_p_cov_imu_world = np.loadtxt(f, delimiter=",")
        net_v_cov_imu_world = net_v_p_cov_imu_world[:, 1:4]
        net_p_cov_imu_world = net_v_p_cov_imu_world[:, 4:7]

        update_acc_freq = 1
        v_update_step = np.searchsorted(ts, net_v_v_cov_t_imu_world)
        p_update_step = np.searchsorted(ts, net_v_v_cov_t_imu_world)

        dcm_imu = Rotation.from_quat(net_q_imu[:, [1, 2, 3, 0]]).as_matrix()

        b, a = signal.butter(3, 0.05, 'lowpass')
        net_gyr_imu = signal.filtfilt(b, a, net_gyr_imu, axis=0)
        net_gyr_gra = np.einsum('tip,tp->ti', dcm_imu, net_gyr_imu)

        net_alpha = np.diff(net_gyr_imu, axis=0)
        net_alpha = np.append(net_alpha[0:1, :], net_alpha, axis=0) / dt
        b, a = signal.butter(3, 0.02, 'lowpass')
        net_alpha = signal.filtfilt(b, a, net_alpha, axis=0)

        gyr_gra = net_gyr_gra
        gyr_imu = net_gyr_imu
        alpha_imu = net_alpha

        trans_gra = np.matmul(dcm_imu, optim_trans)
        gt_v_rigid_world = gt_v + np.cross(gyr_gra, trans_gra)
        v_rigid_world_init = gt_v[start_step]

        gt_p_rigid_world = gt_p + trans_gra
        p_rigid_world_init = gt_p[start_step]

        net_v_rigid_world = net_v_imu_world + np.cross(gyr_gra[v_update_step], trans_gra[v_update_step])
        net_p_rigid_world = net_p_imu_world + trans_gra[v_update_step]
        gt_acc_rigid_body = np.einsum('ip,tp->ti', optim_rot_dcm.T, spline_imu_a + np.cross(gt_gyr, np.cross(gt_gyr, optim_trans)) + np.cross(gt_alpha,optim_trans))

        tao_init = args.tao_init
        drag_init = args.drag_init
        trans_init = args.trans_init # 在imu系下看，imu到rigid的平移
        rot_init = args.rot_init
        rot_dcm = Rotation.from_quat(rot_init[[1,2,3,0]]).as_matrix()

        p_cov_init = args.p_cov_init
        v_cov_init = args.v_cov_init
        tao_cov_init = args.tao_cov_init
        drag_cov_init = args.drag_cov_init
        rot_cov_init = args.rot_cov_init
        trans_cov_init = args.trans_cov_init

        if STATE == 'p_v_k_drag_rot_trans':
            state_all = [np.concatenate(
                (p_rigid_world_init, v_rigid_world_init, tao_init, drag_init, rot_init, trans_init), axis=0)]
            state_cov = block_diag(np.diag(p_cov_init), np.diag(v_cov_init), np.diag(tao_cov_init),
                                   np.diag(drag_cov_init), np.diag(rot_cov_init), np.diag(trans_cov_init))
            state_cov_all = [state_cov]
            ekf = Translation_EKF(p_rigid_world_init, v_rigid_world_init, tao_init, drag_init, rot_init, trans_init,
                              state_cov=state_cov, state_num=10 + 4 + 3)

        buffer_q_rigid = []
        buffer_v_rigid_body = []
        buffer_gyr_rigid = []
        buffer_rpm = []
        out_net_d_all = []
        out_net_d_cov_all = []
        buffer_v_imu = []
        buffer_dt = []

        v_pred_imu_all = []
        p_pred_imu_all = []

        b, a = signal.butter(3, 0.05, 'lowpass')
        net_acc_imu = signal.filtfilt(b, a, net_acc_imu, axis=0)

        for step in tqdm(range(start_step, ts.shape[0])):
            obs_a, obs_v, obs_p = None, None, None
            if use_net_d:
                if (step - start_step) < 0.05 * args.imu_freq:
                    net_dyn_output = np.array([0,0,0])
                    Q = np.diag([1e-2, 1e-2, 1e-2])
                else:
                    if (step - start_step) % 1 == 0 :
                        feat_w = torch.FloatTensor(buffer_gyr_rigid)
                        feat_v_body = torch.FloatTensor(buffer_v_rigid_body)
                        feat_rpm = torch.FloatTensor(buffer_rpm)
                        feature_d = torch.cat((feat_w, feat_v_body, feat_rpm), dim=1).unsqueeze(0).permute( 0, 2, 1).to("cuda")
                        out_net_d, out_net_d_cov = network_d(feature_d)
                        net_dyn_output = torch_to_numpy(out_net_d[0])
                        Q = args.acc_factor * np.diag(torch_to_numpy(torch.exp(2 * out_net_d_cov))[0])

                    else:
                        net_dyn_output = out_net_d_all[-1]
                        Q = out_net_d_cov_all[-1]

                    buffer_gyr_rigid.pop(0)
                    buffer_v_rigid_body.pop(0)
                    buffer_q_rigid.pop(0)
                    buffer_rpm.pop(0)
            else:
                net_dyn_output = np.array([0, 0, 0])
                Q = np.diag([1e-2, 1e-2, 1e-2])

            out_net_d_all.append(net_dyn_output)
            out_net_d_cov_all.append(Q)

            if update_a is True and step % update_acc_freq == 0 and step != start_step:
                obs_a = net_acc_imu[step]
            if update_v is True and step in v_update_step:
                obs_v = net_v_imu_world[np.searchsorted(v_update_step, step)]
                R_v = v_factor * net_v_cov_imu_world[np.searchsorted(v_update_step, step)]
            if update_p is True and step in p_update_step:
                obs_p = net_p_imu_world[np.searchsorted(p_update_step, step)]
                R_p = p_factor * net_p_cov_imu_world[np.searchsorted(p_update_step, step)]

            R_a =  args.obs_acc_cov
            if obs_p is None and obs_v is None and obs_a is not None:
                R = np.diag([R_a[0], R_a[1], R_a[2]])
            elif obs_p is not None and obs_v is not None and obs_a is None:
                R = np.diag([R_p[0], R_p[1], R_p[2], R_v[0], R_v[1], R_v[2]])
            elif obs_p is None and obs_v is not None and obs_a is None:
                R = np.diag([R_v[0], R_v[1], R_v[2]])
            elif obs_p is None and obs_v is not None and obs_a is not None:
                R = np.diag([R_v[0], R_v[1], R_v[2], R_a[0], R_a[1], R_a[2]])
            elif obs_p is not None and obs_v is None and obs_a is None:
                R = np.diag([R_p[0], R_p[1], R_p[2]])
            elif obs_p is not None and obs_v is None and obs_a is not None:
                R = np.diag([R_p[0], R_p[1], R_p[2], R_a[0], R_a[1], R_a[2]])
            elif obs_p is not None and obs_v is not None and obs_a is not None:
                R = np.diag([R_p[0], R_p[1], R_p[2], R_v[0], R_v[1], R_v[2], R_a[0], R_a[1], R_a[2]])
            else:
                R = None

            state, state_cov = ekf.compute_all(dcm_imu=dcm_imu[step], dt=dt[step],
                                               rpm=meas_rpm[step], R=R, w=gyr_imu[step],
                                               alpha=alpha_imu[step], obs_a=obs_a, obs_v=obs_v,
                                               obs_p=obs_p, net_dyn_output=net_dyn_output, Q = Q)

            dcm_rigid = np.matmul(dcm_imu[step],Rotation.from_quat(state[[11, 12, 13, 10]]).as_matrix())
            dcm_rigid_all.append(dcm_rigid)
            trans_gra = np.matmul(dcm_imu[step], state[14:17])
            v_pred_imu = state[3:6] - np.cross(gyr_gra[step], trans_gra)
            v_pred_imu_all.append(v_pred_imu)
            p_pred_imu = state[0:3] - trans_gra
            p_pred_imu_all.append(p_pred_imu)
            net_gyr_rigid = rot_dcm.T @ gyr_imu[step]

            state_all.append(state)
            state_cov_all.append(state_cov)
            q_rigid = Rotation.from_matrix(dcm_rigid).as_quat()
            q_rigid = q_rigid[[3, 0, 1, 2]]
            v_rigid_body.append(dcm_rigid.T @ state[3:6])
            if use_net_d:
                buffer_q_rigid.append(q_rigid)
                buffer_v_rigid_body.append(dcm_rigid.T @ state[3:6])
                buffer_v_imu.append(v_pred_imu)
                buffer_gyr_rigid.append(net_gyr_rigid)
                buffer_rpm.append(meas_rpm[step])
                buffer_dt.append(dt[step])

        state_all = np.array(state_all[1:])  # 第一个去除
        state_cov_all = np.array(state_cov_all[1:])
        v_pred_imu_all = np.array(v_pred_imu_all)
        p_pred_imu_all = np.array(p_pred_imu_all)

        print(data_path)
        a_rigid_error.append(np.mean(np.abs(np.array(ekf.net_d_acc_all) - gt_acc_rigid_body),axis=0))
        print('a_rigid_error: ', np.mean(np.abs(np.array(ekf.net_d_acc_all) - gt_acc_rigid_body),axis=0))

        xlb = 'ts'
        ylbs = ['x', 'y', 'z']
        # p
        dpi = 90
        figsize = (16, 9)
        fig1 = plt.figure(dpi=dpi, figsize=figsize)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts[start_step:],state_all[:, i])
            plt.plot(ts[start_step:],gt_p_rigid_world[start_step:, i])
            plt.plot(ts[p_update_step[np.searchsorted(p_update_step, start_step):]],
                     net_p_rigid_world[np.searchsorted(p_update_step, start_step):, i])
            plt.legend(['ekf_p_rigid', 'gt_p_rigid', 'only_net_p_rigid'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)

        p_rigid_error.append(np.mean(np.abs(state_all[:, 0:3 ] - gt_p_rigid_world[start_step:, 0:3]), axis=0))
        print('p_rigid_error: ', np.mean(np.abs(state_all[:, 0:3 ] - gt_p_rigid_world[start_step:, 0:3]), axis=0))

        dpi = 90
        figsize = (16, 9)
        fig4 = plt.figure(dpi=dpi, figsize=figsize)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts[start_step:],p_pred_imu_all[:, i])
            plt.plot(ts[start_step:],gt_p[start_step:, i])
            plt.plot(ts[p_update_step[np.searchsorted(p_update_step, start_step):]],
                     net_p_imu_world[np.searchsorted(p_update_step, start_step):, i])
            plt.legend(['ekf_p_imu', 'gt_p_imu', 'only_net_p_imu'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)
        print("p_imu_error: ", np.mean(np.abs(gt_p[start_step:, ] - p_pred_imu_all), axis=0))

        p_error.append(np.mean(np.abs(gt_p[start_step:, ] - p_pred_imu_all), axis=0))
        # v
        dpi = 90
        figsize = (16, 9)
        fig2 = plt.figure(dpi=dpi, figsize=figsize)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts[start_step:],state_all[:, i + 3])
            plt.plot(ts[start_step:],gt_v_rigid_world[start_step:, i])
            plt.plot(ts[v_update_step[np.searchsorted(v_update_step, start_step):]],
                     net_v_rigid_world[np.searchsorted(v_update_step, start_step):, i])
            plt.legend(['ekf_v_rigid', 'gt_v_rigid', 'only_net_v_rigid'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)

        v_rigid_error.append(np.mean(np.abs(gt_v_rigid_world[start_step:, ] - state_all[:, 3:6 ]), axis=0))
        print('v_rigid_error: ',np.mean(np.abs(gt_v_rigid_world[start_step:, ] - state_all[:, 3:6 ]), axis=0) )

        v_rigid_body_error.append(np.mean(np.abs(gt_v_rigid_body[start_step:, ] - np.array(v_rigid_body)), axis=0))
        print('v_rigid_body_error: ', np.mean(np.abs(gt_v_rigid_body[start_step:, ] - np.array(v_rigid_body)), axis=0))

        dpi = 90
        figsize = (16, 9)
        fig5 = plt.figure(dpi=dpi, figsize=figsize)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts[start_step:],v_pred_imu_all[:, i])
            plt.plot(ts[start_step:],gt_v[start_step:, i])
            plt.plot(ts[v_update_step[np.searchsorted(v_update_step, start_step):]],
                     net_v_imu_world[np.searchsorted(v_update_step, start_step):, i])
            plt.legend(['ekf_v_imu', 'gt_v_imu', 'only_net_v_imu'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)

        print("v_imu_error: ", np.mean(np.abs(gt_v[start_step:] - v_pred_imu_all), axis=0))

        v_error.append(np.mean(np.abs(gt_v[start_step:] - v_pred_imu_all), axis=0))

        # k, drag
        dpi = 90
        figsize = (16, 9)
        fig3 = plt.figure(dpi=dpi, figsize=figsize)
        for i in range(4):
            plt.subplot(4, 1, i + 1)
            plt.plot(ts[start_step:],state_all[:, i + 6])
            plt.legend(['tao', 'drag_x', 'drag_y', 'drag_z'])
        plt.xlabel(xlb)

        # rot
        dpi = 90
        figsize = (16, 9)
        fig6 = plt.figure(dpi=dpi, figsize=figsize)
        rot_euler = Rotation.from_quat(state_all[:, 10:14][:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts[start_step:],rot_euler[:, i])
            plt.legend(['rot_roll', 'rot_pitch', 'rot_yaw'])
        plt.xlabel(xlb)

        # trans
        dpi = 90
        figsize = (16, 9)
        fig7 = plt.figure(dpi=dpi, figsize=figsize)
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts[start_step:],state_all[:, i + 14])
            plt.legend(['trans_x', 'trans_y', 'trans_z'])
        plt.xlabel(xlb)

        if args.show_figure:
            fig1.show(), fig4.show(), fig2.show(), fig5.show(), fig3.show(), fig6.show(), fig7.show()
        if args.save_result:
            fig1.savefig(file_name + '/p_rigid_world/' + data_path + '_p_rigid_world.png')
            fig4.savefig(file_name + '/p_imu_world/' + data_path + '_p_imu_world.png')
            fig2.savefig(file_name + '/v_rigid_world/' + data_path + '_v_rigid_world.png')
            fig5.savefig(file_name + '/v_imu_world/' + data_path + '_v_imu_world.png')
            fig3.savefig(file_name + '/tao_d/' + data_path + '_tao_d.png')
            fig6.savefig(file_name + '/rot/' + data_path + '_rot.png')
            fig7.savefig(file_name + '/trans/' + data_path + '_trans.png')
        plt.close("all")

        output_data = np.concatenate(
            (ts.reshape(-1, 1), net_q_imu, p_pred_imu_all, v_pred_imu_all, state_all[:,0:3], state_all[:,3:6],v_rigid_body,np.array(ekf.net_d_acc_all)), axis=1)
        np.savetxt(file_name + '/result/' + data_path + '.txt', output_data, delimiter=',')


    p_error = np.array(p_error)
    p_rigid_error = np.array(p_rigid_error)
    v_error = np.array(v_error)
    v_rigid_error = np.array(v_rigid_error)
    a_rigid_error = np.array(a_rigid_error)
    v_rigid_body_error = np.array(v_rigid_body_error)

    p_error_mean = np.mean(p_error, axis=0)
    p_rigid_error_mean = np.mean(p_rigid_error, axis=0)
    v_error_mean = np.mean(v_error, axis=0)
    v_rigid_error_mean = np.mean(v_rigid_error, axis=0)
    a_rigid_error_mean = np.mean(a_rigid_error, axis=0)
    v_rigid_body_error_mean = np.mean(v_rigid_body_error, axis=0)

    np.savetxt(file_name + '/p_error_' + str(p_error_mean) + '.txt', p_error, delimiter=',')
    np.savetxt(file_name + '/v_error_' + str(v_error_mean) + '.txt', v_error, delimiter=',')
    np.savetxt(file_name + '/p_rigid_error_' + str(p_rigid_error_mean) + '.txt', p_rigid_error, delimiter=',')
    np.savetxt(file_name + '/v_rigid_error_' + str(v_rigid_error_mean) + '.txt', v_rigid_error, delimiter=',')
    np.savetxt(file_name + '/a_rigid_error_' + str(a_rigid_error_mean) + '.txt', a_rigid_error, delimiter=',')
    np.savetxt(file_name + '/v_rigid_body_error_' + str(v_rigid_body_error_mean) + '.txt', v_rigid_body_error,delimiter=',')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # network result path
    parser.add_argument("--network_dyn_path", type=str, default= "../../Res_dynamic/train_output/checkpoints/checkpoint_1113.pt")
    parser.add_argument("--network_v_p_path", type=str,default="../../V_P_net/train_output/vp/")
    parser.add_argument("--network_acc_path", type=str,default="../../Rotation_ekf/output/net_acc/")
    parser.add_argument("--network_gyr_path", type=str,default="../../Rotation_ekf/output/net_gyr/")
    parser.add_argument("--network_q_path", type=str,default="../../Rotation_ekf/output/ekf_q/")

    # dataset
    parser.add_argument("--test_list", type=str, default="../../dataset/test.txt")
    parser.add_argument("--out_dir", type=str, default="../output")
    parser.add_argument(
        "--root_dir", type=str, default="../../dataset", help="Path to data directory")

    parser.add_argument("--win_size", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1256)
    parser.add_argument("--imu_freq", type=int, default=400)
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")

    # dynamic_param_init
    parser.add_argument("--tao_init", default=np.array([1.1]))
    parser.add_argument("--drag_init", default=np.array([0.0, 0.0, 0.0]))
    parser.add_argument("--trans_init", default=np.array([0., 0., 0]) )
    parser.add_argument("--rot_init", default=np.array([1,0,0,0]))

    # init_cov
    parser.add_argument("--p_cov_init", default=np.array([1e-4, 1e-4, 1e-4]))
    parser.add_argument("--v_cov_init", default=np.array([1e-6, 1e-6, 1e-6]))
    parser.add_argument("--tao_cov_init", default=np.array([1e-4]))
    parser.add_argument("--drag_cov_init", default=np.array([5e-4, 5e-4, 5e-4]))
    parser.add_argument("--rot_cov_init", default=np.array([5e-5, 5e-5, 5e-5]))
    parser.add_argument("--trans_cov_init", default=np.array([5e-4, 5e-4, 5e-4]))

    # obs_cov
    parser.add_argument("--obs_acc_cov", default=np.array([1e-1, 1e-1, 1e-1]))
    parser.add_argument("--network_v_p_cov_path", type=str,default="../../V_P_net/train_output/vp_cov/")

    # tuning factor
    parser.add_argument("--v_factor", default=np.array([1, 1, 1]))
    parser.add_argument("--p_factor", default=np.array([1e-1, 1e-1, 1e-1]))
    parser.add_argument("--acc_factor", default= 10 )

    args = parser.parse_args()

    translation_ekf( update_a = True, use_net_d=True, update_v = True, update_p = True, args=args)
    # translation_ekf(update_a=True, use_net_d=True, update_v=True, update_p=False, exp_name="/update_vad/")
    # translation_ekf(update_a=True, use_net_d=True, update_v=False, update_p=False, exp_name="/update_ad/")
    # translation_ekf(update_a=False, use_net_d=True, update_v=False, update_p=False, exp_name="/update_d/")
    # translation_ekf(update_a=False, use_net_d=False, update_v=False, update_p=False, exp_name="/update_none/")
    # translation_ekf(update_a = True, use_net_d=False, update_v=True, update_p=True, exp_name="/update_pva")
    # translation_ekf(update_a = False, use_net_d=True, update_v=False, update_p=True, exp_name="/update_pd")
    # translation_ekf(update_a=False, use_net_d=True, update_v=True, update_p=False, exp_name="/update_vd")
    # translation_ekf(update_a=True, use_net_d=True, update_v=False, update_p=True, exp_name="/update_pad")
    # translation_ekf(update_a=False, use_net_d=True, update_v=True, update_p=True, exp_name="/update_pvd")
    # translation_ekf(update_a=False, use_net_d=False, update_v=True, update_p=True, exp_name="/update_pv")
    # translation_ekf(update_a=False, use_net_d=False, update_v=False, update_p=True, exp_name="/update_p")
    # translation_ekf(update_a=False, use_net_d=False, update_v=True, update_p=False, exp_name="/update_v")
    # translation_ekf(update_a=True, use_net_d=False, update_v=False, update_p=False, exp_name="/update_a")

