import time
import torch
import tqdm

from network.model_factory import get_model
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pyquaternion import Quaternion
from gravity_align_ekf import gravity_align_EKF
import argparse
import os
from os import path as osp

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

def q_integrate(gyr, ts_win, q0):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    feat_gyr = gyr
    dalte_w = (feat_gyr[1:] + feat_gyr[:-1]) / 2

    dalte_gyr_norm = np.linalg.norm(dalte_w, ord=2, axis=1, keepdims=True)
    dalte_intint = dalte_gyr_norm * np.expand_dims(ts_win[0:], 1) / 2  # 因为我输入的是dt，所以这边原本的ts_win[1:]改成了ts_win[0:]
    w_point = dalte_w / dalte_gyr_norm
    dalte_q_w = np.cos(dalte_intint)
    dalte_q_xyz = w_point * np.sin(dalte_intint)
    dalte_q_wxyz = np.concatenate((dalte_q_w, dalte_q_xyz), axis=1)

    dalte_q_1 = Quaternion(q0)
    dalte_q_2 = Quaternion(dalte_q_wxyz[0])
    dalte_q_x = dalte_q_1 * dalte_q_2
    dalte_q_winst = np.expand_dims(dalte_q_1.q, axis=0)
    dalte_q_win = np.concatenate((dalte_q_winst, np.expand_dims(dalte_q_x.q, axis=0)), axis=0)
    for iii in range(1, len(dalte_q_wxyz[:, 0])):
        dalte_q_x = dalte_q_x * Quaternion(dalte_q_wxyz[iii])
        if dalte_q_x.w < 0:
            dalte_q_x.q = - dalte_q_x.q
        dalte_q_win = np.concatenate((dalte_q_win, np.expand_dims(dalte_q_x.q, axis=0)), axis=0)

    dalte_q_x_xnorm = dalte_q_x.normalised
    dalte_q_diff = dalte_q_x_xnorm.q
    dalte_q_diff = np.array(dalte_q_diff)
    dalte_q_win = np.array(dalte_q_win)

    return dalte_q_diff, dalte_q_win

def change_yaw(euler):
    th = 330
    d_euler = np.diff(euler, axis=0, prepend=euler[0].reshape(1, 3))
    for i in np.where(d_euler[:, 2] < -th)[0]:
        euler[i:, 2] += 2 * 180
    for i in np.where(d_euler[:, 2] > th)[0]:
        euler[i:, 2] -= 2 * 180
    return euler

def ROTATION_EKF(args):

    start_step = 0

    imu_freq = args.imu_freq
    root_dir = args.root_dir
    out_dir = args.out_dir
    test_path = args.test_list
    data_path_s = get_datalist(test_path)
    for data_path in tqdm.tqdm(data_path_s):
        with h5py.File(root_dir + '/' + data_path + '/data.hdf5', "r") as f:
            ts = np.copy(f["ts"])  # timestamp
            gt_q = np.copy(f["gt_q"])  # quaternion of body frame in world frame

        dt = np.diff(ts)
        dt = np.append(dt[0], dt).reshape(-1, 1)

        with open(osp.join(args.network_gyr_out_path + data_path + "_gyr.txt"), encoding='utf-8') as f:
            pred_gyr_all = np.loadtxt(f, delimiter=",")
        with open(osp.join(args.network_acc_out_path + data_path + "_acc.txt"), encoding='utf-8') as f:
            pred_acc_all = np.loadtxt(f, delimiter=",")

        _, pred_q = q_integrate(pred_gyr_all, dt[start_step + 1:, 0],gt_q[start_step])

        euler_gt_q = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)
        euler_pred = Rotation.from_quat(pred_q[:, [1, 2, 3, 0]]).as_euler('xyz', degrees=True)

        acc_cov = np.expand_dims([0.01], axis=1).repeat(pred_acc_all.shape[0], axis=0)
        ekf = gravity_align_EKF(gyr=(pred_gyr_all[1:, :] + pred_gyr_all[:-1, :]) / 2, acc=pred_acc_all,
                                       acc_cov=acc_cov ** 2,
                                       noises=[0.05 ** 2],  # big是0.05
                                       frequency=imu_freq,
                                       dt_seq=dt[start_step + 1:, 0],  # 中值积分 + 1
                                       update_rate=10,
                                       arch='origin',
                                       Init_state_cov=0.0000000001,
                                       q0=gt_q[start_step], frame='ENU')

        ekf_q = ekf.Q / np.linalg.norm(ekf.Q, axis=1, keepdims=True)
        ekf_q[ekf_q[:, 0] < 0] = (-1) * ekf_q[ekf_q[:, 0] < 0]
        ekf_euler = Rotation.from_quat(ekf_q[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=True)

        euler_pred = change_yaw(euler_pred)
        euler_gt_q = change_yaw(euler_gt_q)
        ekf_euler = change_yaw(ekf_euler)

        xlb = 'ts'
        ylbs = ['roll', 'pitch', 'yaw']

        dpi = 90
        figsize = (16, 9)
        fig1 = plt.figure(dpi=dpi, figsize=figsize)
        fig1.suptitle('error euler')
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts,np.abs(euler_pred[:, i] - euler_gt_q[start_step :, i]))
            plt.plot(ts,np.abs(ekf_euler[:, i] - euler_gt_q[start_step :, i]))
            plt.legend(['error_pred_euler', 'error_ekf_euler'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)

        dpi = 90
        figsize = (16, 9)
        fig2 = plt.figure(dpi=dpi, figsize=figsize)
        fig2.suptitle('euler')
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(ts,euler_pred[:, i])
            plt.plot(ts,euler_gt_q[start_step:, i])
            plt.plot(ts,ekf_euler[:, i])
            plt.legend(['euler_pred', 'euler_gt_euler', 'ekf_euler'])
            plt.ylabel(ylbs[i])
        plt.xlabel(xlb)

        if args.save_result:
            file_names = ['net_gyr','net_acc','ekf_q','error_euler','euler']
            for file_name in file_names:
                if not osp.isdir( out_dir + '/'+file_name+'/'):
                    os.makedirs(out_dir + '/'+file_name+'/')
                    print('create '+ out_dir + '/'+file_name+'/')
            np.savetxt(out_dir+'/net_gyr/' + data_path + '_gyr.txt', pred_gyr_all, delimiter=',')
            np.savetxt(out_dir+'/net_acc/' + data_path + '_acc.txt', pred_acc_all, delimiter=',')
            np.savetxt(out_dir+'/ekf_q/'+data_path + '_ekf_q.txt', ekf_q, delimiter=',')
            fig1.savefig(out_dir+'/error_euler/'+data_path + '_end_error.png')
            fig2.savefig(out_dir+'/euler/'+data_path + '_end.png')

        if args.show_figure:
            fig1.show()
            fig2.show()

        print(data_path)

    print('a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_acc_out_path", type=str, default= "../output/net_acc/")
    parser.add_argument("--network_gyr_out_path", type=str, default= "../output/net_gyr/")
    parser.add_argument("--win_size", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1256)
    parser.add_argument("--imu_freq", type=int, default=400)
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_list", type=str, default="../../dataset/test.txt")
    parser.add_argument("--out_dir", type=str, default="../output")
    parser.add_argument(
        "--root_dir", type=str, default="../../dataset", help="Path to data directory")

    args = parser.parse_args()

    ROTATION_EKF(args)
