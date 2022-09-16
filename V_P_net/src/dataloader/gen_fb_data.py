#!/usr/bin/env python3

"""
gen_fb_data.py

Input (FB dataset raw): my_timestamps_p.txt, calib_state.txt, imu_measurements.txt, evolving_state.txt, attitude.txt
Output: data.hdf5
    - ts
    - raw acc and gyr measurements
    - gt-calibrated acc and gyr measurements
    - ground truth (gt) states (R, p, v)
    - integration rotation (with offline calibration)
    - attitude filter rotation
    - offline calibration parameters

Note: the dataset has been processed in a particular way such that the time difference between ts is almost always 1ms. This is for the training and testing the network. 
Image frequency is known, and IMU data is interpolated evenly between the two images.
"""

import os
import sys
from os import path as osp

import h5py
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from numba import jit
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
sys.path.append('/home/jiangcx/桌面/TLIO/DL_IMU/src')
from utils.math_utils import mat_exp, unwrap_rpy, wrap_rpy


def imu_integrate(gravity, last_state, imu_data, dt):
    """
    Given compensated IMU data and corresponding dt, propagate the pregtus state in the world frame
    """
    last_r = Rotation.from_rotvec(last_state[0:3])
    last_p = last_state[3:6]
    last_v = last_state[6:9]
    acc = imu_data[:3]
    omega = imu_data[3:]

    last_R = last_r.as_matrix()
    dR = mat_exp(omega * dt)

    new_R = last_R.dot(dR)
    new_v = last_v + gravity * dt + last_R.dot(acc * dt)
    new_p = (
        last_p
        + last_v * dt
        + 0.5 * gravity * dt * dt
        + 0.5 * last_R.dot(acc * dt * dt)
    )
    new_r = Rotation.from_matrix(new_R)

    return np.concatenate((new_r.as_rotvec(), new_p, new_v), axis=0)

blackbird = True

def save_hdf5(args):

    # get list of data to process
    f = open(args.data_list, "r")
    name = [line.rstrip() for line in f]
    f.close()
    n_data = len(name)
    print(f"total {n_data} datasets")

    gravity = np.array([0, 0, -args.gravity])

    for i in progressbar.progressbar(range(n_data), redirect_stdout=True):
        n = name[i]

        # set flag for which data has been processed
        datapath = osp.join(args.data_dir, n)
        flag_file = os.path.join(args.data_dir, n, "hey.txt")
        if os.path.exists(flag_file):
            print(f"{i}th data {n} processed, skip")
            continue
        else:
            print(f"processing data {i} - {n}")
            # f = open(flag_file, 'w'); f.write('hey'); f.close()

        # start with the 20th image processed, bypass gt initialization
        image_ts = np.loadtxt(osp.join(datapath, "my_timestamps_p.txt"))
        imu_meas = np.loadtxt(osp.join(datapath, "imu_measurements.txt"), delimiter=",")
        gt_states = np.loadtxt(osp.join(datapath, "evolving_state.txt"), delimiter=",")
        if blackbird:
            targ_rpm = np.loadtxt(osp.join(datapath, "targ_rpm.txt"), delimiter=",")
            meas_rpm = np.loadtxt(osp.join(datapath, "meas_rpm.txt"), delimiter=",")

        # calib_data = np.loadtxt(osp.join(datapath, "calib_state.txt"), delimiter=",")
        # attitudes = np.loadtxt(
        #     osp.join(datapath, "atttitude.txt"), delimiter=",", skiprows=3
        # )

        # find initial state, start from the 21st output from gt_state
        # start_t = image_ts[20]

        # plt.figure()
        # for i in range(3):
        #     plt.subplot(3,1,i+1)
        #     plt.plot(gt_states[:,5+i])
        # plt.show()
        # plt.figure()
        # for i in range(3):
        #     plt.subplot(3,1,i+1)
        #     plt.plot(gt_states[:,8+i])
        # plt.show()
        # plt.figure()
        # for i in range(3):
        #     plt.subplot(3,1,i+1)
        #     plt.plot(gt_states[:,8+i])
        # plt.show()

        start_t = image_ts[0] # blackbird_cut
        imu_idx = np.searchsorted(imu_meas[:, 0], start_t)
        gt_idx = np.searchsorted(gt_states[:, 0], start_t)
        if blackbird:
            targ_rpm_idx = np.searchsorted(targ_rpm[:, 0], start_t)
            meas_rpm_idx = np.searchsorted(meas_rpm[:, 0], start_t)


        # get imu_data - raw and calibrated
        print("obtain raw and gt-calibrated IMU data")
        imu_data = imu_meas[imu_idx:, :]
        ts = imu_data[:, 0] * 1e-6  # s
        acc = imu_data[:, 1:4]  # raw
        gyr = imu_data[:, 7:10]
        acc = imu_data[:, 4:7]  # calibrated with gt calibration
        gyr = imu_data[:, 10:13]


        # Ind1 Below code is used to generate X_state data which is our evolving_data.txt
        # get state_data (same timestamps as imu_data), integrated with gt-calibrated IMU
        # 从第20帧的gt得到的初始状态（可以看做是真值的状态）
        print("obtain gt states by integrating gt-calibrated IMU")
        N = imu_data.shape[0]
        state_data = np.zeros((N, 9))
        if blackbird:
            targ_rpm_data = np.zeros((N, 4))
            meas_rpm_data = np.zeros((N, 4))

        r_init = Rotation.from_quat(
            [
                gt_states[gt_idx, 2],
                gt_states[gt_idx, 3],
                gt_states[gt_idx, 4],
                gt_states[gt_idx, 1],
            ]
        )
        p_init = [
            gt_states[gt_idx, 5],
            gt_states[gt_idx, 6],
            gt_states[gt_idx, 7],
        ]
        v_init = [
            gt_states[gt_idx, 8],
            gt_states[gt_idx, 9],
            gt_states[gt_idx, 10],
        ]
        state_init = np.concatenate((r_init.as_rotvec(), p_init, v_init), axis=0)
        state_data[0, :] = state_init
        if blackbird:
            targ_rpm_data[0,:] = targ_rpm[targ_rpm_idx,1:]
            meas_rpm_data[0, :] = meas_rpm[meas_rpm_idx, 1:]


        # 根据imu和中间的gt真值，积分得到真实轨迹
        for i in progressbar.progressbar(range(1, N)):
            # get calibrated imu data for integration
            imu_data_i = np.concatenate((imu_data[i, 4:7], imu_data[i, 10:13]), axis=0)
            curr_t = imu_data[i, 0]
            past_t = imu_data[i - 1, 0]
            dt = (curr_t - past_t) * 1e-6  # s

            last_state = state_data[i - 1, :]
            new_state = imu_integrate(gravity, last_state, imu_data_i, dt)
            state_data[i, :] = new_state

            # if this state has gt output, correct with gt
            has_gt = imu_data[i, 13]
            if has_gt == 1:
                gt_idx = np.searchsorted(gt_states[:, 0], imu_data[i, 0])
                if blackbird:
                    targ_rpm_idx = np.searchsorted(targ_rpm[:, 0], imu_data[i, 0])
                    meas_rpm_idx = np.searchsorted(meas_rpm[:, 0], imu_data[i, 0])

                try:
                    r_gt = Rotation.from_quat(
                        [
                            gt_states[gt_idx, 2], # x
                            gt_states[gt_idx, 3], # y
                            gt_states[gt_idx, 4], # z
                            gt_states[gt_idx, 1], # w
                        ]
                    )
                except:
                    plt.plot(gt_states[:, 0])
                    plt.plot(imu_data[:, 0])
                    plt.show()
                    print('aa')

                p_gt = [
                    gt_states[gt_idx, 5],
                    gt_states[gt_idx, 6],
                    gt_states[gt_idx, 7],
                ]
                v_gt = [
                    gt_states[gt_idx, 8],
                    gt_states[gt_idx, 9],
                    gt_states[gt_idx, 10],
                ]
                gt_state = np.concatenate((r_gt.as_rotvec(), p_gt, v_gt), axis=0)
                state_data[i, :] = gt_state

                if blackbird:
                    try:
                        targ_rpm_data[i,:] = targ_rpm[targ_rpm_idx,1:]
                    except:
                        print('aa')
                    meas_rpm_data[i,:] = meas_rpm[meas_rpm_idx,1:]


        # adding timestamps in state_data
        state_data = np.concatenate(
            (np.expand_dims(imu_data[:, 0], axis=1), state_data), axis=1
        )
        # Ind1 Above code is used to generate X_state data which is our evolving_data.txt

        # gt data
        gt_rvec = state_data[:, 1:4] # quaternion
        gt_p = state_data[:, 4:7]    # position
        gt_v = state_data[:, 7:10]   # velocity
        gt_r = Rotation.from_rotvec(gt_rvec)
        gt_q = gt_r.as_quat()
        gt_q = np.concatenate(
            [np.expand_dims(gt_q[:, 3], axis=1), gt_q[:, 0:3]], axis=1
        )
        # 又转换为了四元数

        # # ljh for testing 
        # np.savetxt('/home/ljh/dataset/TLIO_data/gt_p.csv',gt_p, delimiter=',')
        # np.savetxt('/home/ljh/dataset/TLIO_data/gt_v.csv',gt_v, delimiter=',')

        # # Ind2 Below Don't how to use this code, length of atttitude.txt's line is not right
        # # get offline and factory calibration
        # print("obtain offline and factory calibration")
        # with open(osp.join(datapath, "atttitude.txt"), "r") as f:
        #     line = f.readline()
        #     line = f.readline()
        # init_calib = np.fromstring(line, sep=",")
        # # init_calib_fac = None
        # accScaleInv = init_calib[1:10].reshape((3, 3))
        # gyroScaleInv = init_calib[10:19].reshape((3, 3))
        # gyroGSense = init_calib[19:28].reshape((3, 3))
        # accBias = init_calib[28:31].reshape((3, 1))
        # gyroBias = init_calib[31:34].reshape((3, 1))
        # offline_calib = np.concatenate(
        #     (
        #         accScaleInv.flatten(),
        #         gyroScaleInv.flatten(),
        #         gyroGSense.flatten(),
        #         accBias.flatten(),
        #         gyroBias.flatten(),
        #     )
        # )
        
        # # calibrate raw IMU with fixed calibration
        # print("calibrate IMU with fixed calibration")
        # acc_calib = (np.dot(accScaleInv, acc.T) - accBias).T
        # gyro_calib = (
        #     np.dot(gyroScaleInv, gyr.T)
        #     - np.dot(gyroGSense, acc.T)
        #     - gyroBias
        # ).T


        # # integrate using fixed calibration data
        # print("integrate R using fixed calibrated data")
        # N = ts.shape[0]
        # rvec_integration = np.zeros((N, 3))
        # rvec_integration[0, :] = state_data[0, 1:4]

        # for i in progressbar.progressbar(range(1, N)):
        #     dt = ts[i] - ts[i - 1]
        #     last_rvec = Rotation.from_rotvec(rvec_integration[i - 1, :])
        #     last_R = last_rvec.as_matrix()
        #     omega = gyro_calib[i, :]
        #     dR = mat_exp(omega * dt)
        #     next_R = last_R.dot(dR)
        #     next_r = Rotation.from_matrix(next_R)
        #     next_rvec = next_r.as_rotvec()
        #     rvec_integration[i, :] = next_rvec
        # integration_r = Rotation.from_rotvec(rvec_integration)
        # integration_q = integration_r.as_quat()
        # integration_q_wxyz = np.concatenate(
        #     [np.expand_dims(integration_q[:, 3], axis=1), integration_q[:, 0:3]], axis=1
        # )
        # # # Ind2 ABove Don't how to use this code, length of atttitude.txt's line is not right

        # get attitude filter data // 
        # in order to make sure the X_state data is between vicon/gt start and end key frame
        # print("obtain attitude filter rotation")
        # ts_filter = attitudes[:, 0] * 1e-6  # s
        # ts_start_index = 0
        # ts_end_index = -1
        # if ts_filter[0] > ts[0]:
        #     print("dataset " + n + " has smaller attitude filter timerange")
        #     ts_start_index = np.where(ts > ts_filter[0])[0][0]
        # if ts_filter[-1] < ts[-1]:
        #     print("dataset " + n + " has smaller attitude filter timerange")
        #     ts_end_index = np.where(ts < ts_filter[-1])[0][-1]

        # filter_r = Rotation.from_quat(
        #     np.concatenate(
        #         [attitudes[:, 2:5], np.expand_dims(attitudes[:, 1], axis=1)], axis=1
        #     )
        # )
        # R_filter = filter_r.as_matrix()
        # R_wf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        # R_filter = np.matmul(R_wf, R_filter)
        # filter_r = Rotation.from_matrix(R_filter)  # corrected rotation for filter
        # filter_rpy = filter_r.as_euler("xyz", degrees=True)
        # uw_filter_rpy = unwrap_rpy(filter_rpy)
        # uw_filter_rpy_interp = interp1d(ts_filter, uw_filter_rpy, axis=0)(
        #     ts[ts_start_index:ts_end_index]
        # )
        # filter_rpy_interp = wrap_rpy(uw_filter_rpy_interp)
        # filter_r_interp = Rotation.from_euler("xyz", filter_rpy_interp, degrees=True)
        # filter_q_interp = filter_r_interp.as_quat()
        # filter_q_wxyz_interp = np.concatenate(
        #     [np.expand_dims(filter_q_interp[:, 3], axis=1), filter_q_interp[:, 0:3]],
        #     axis=1,
        # )

        # # truncate to attitude filter range
        # ts = ts[ts_start_index:ts_end_index]
        # acc = acc[ts_start_index:ts_end_index, :]
        # gyr = gyr[ts_start_index:ts_end_index, :]
        # acc = acc[ts_start_index:ts_end_index, :]
        # gyr = gyr[ts_start_index:ts_end_index, :]
        # gt_q = gt_q[ts_start_index:ts_end_index, :]
        # gt_p = gt_p[ts_start_index:ts_end_index, :]
        # gt_v = gt_v[ts_start_index:ts_end_index, :]
        # # integration_q_wxyz = integration_q_wxyz[ts_start_index:ts_end_index, :]
        # if ts.shape[0] != filter_q_wxyz_interp.shape[0]:
        #     print("data and attitude filter time does not match!")

        # 计算输出真值
        
        
        
        # output
        outdir = osp.join(args.output_dir, n)
        if not osp.isdir(outdir):
            os.makedirs(outdir)

        # everything under the same timestamp ts
        with h5py.File(osp.join(outdir, "data.hdf5"), "w") as f:
            ts_s = f.create_dataset("ts", data=ts)
            acc_calibrated_s = f.create_dataset("acc_calibrated", data=acc)
            gyr_calibrated_s = f.create_dataset("gyr_calibrated", data=gyr)
            acc_s = f.create_dataset("acc", data=acc)
            gyr_s = f.create_dataset("gyr", data=gyr)
            gt_p_s = f.create_dataset("gt_p", data=gt_p)
            gt_v_s = f.create_dataset("gt_v", data=gt_v)
            gt_q_s = f.create_dataset("gt_q", data=gt_q)
            if blackbird:
                targ_rpm_s = f.create_dataset("targ_rpm", data=targ_rpm_data)
                meas_rpm_s = f.create_dataset("meas_rpm",data=meas_rpm_data)

            # integration_q_wxyz_s = f.create_dataset(
            #     "integration_q_wxyz", data=integration_q_wxyz
            # )
            # filter_q_wxyz_s = f.create_dataset(
            #     "filter_q_wxyz", data=filter_q_wxyz_interp
            # )
            # offline_calib_s = f.create_dataset("offline_calib", data=offline_calib)
            print("File data.hdf5 written to " + outdir)
            # 最终输出的是 imu的原始量（6） 矫正加速度（6）gt四元数（4）gt_position（3）gt_speed(3)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity", type=float, default=9.81)
    parser.add_argument("--output_dir", type=str, default="/home/jiangcx/桌面/TLIO/DL_IMU/min_snap_data")
    parser.add_argument(
        "--data_dir", type=str, default="/home/jiangcx/桌面/TLIO/DL_IMU/min_snap_data"
    )
    parser.add_argument(
        "--data_list",
        type=str,
        default="/home/jiangcx/桌面/TLIO/DL_IMU/min_snap_data/gen_list.txt",
    )
    args = parser.parse_args()

save_hdf5(args)