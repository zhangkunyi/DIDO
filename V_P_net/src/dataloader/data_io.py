from os import path as osp

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from utils.logging import logging
from utils.math_utils import unwrap_rpy, wrap_rpy


class DataIO:
    def __init__(self):
        # raw dataset - ts in us
        self.ts_all = None
        self.acc_all = None
        self.gyr_all = None
        self.dataset_size = None
        self.init_ts = None
        self.R_init = np.eye(3)
        # gt data
        self.gt_ts = None
        self.gt_p = None
        self.gt_v = None
        self.gt_eul = None
        self.gt_R = None
        self.gt_rq = None
        self.gt_ba = None
        self.gt_bg = None
        # attitude filter data
        self.filter_ts = None
        self.filter_eul = None

    def load_all(self, dataset, args):
        """
        load timestamps, acc and gyr data from dataset
        """
        with h5py.File(osp.join(args.root_dir, dataset, "data.hdf5"), "r") as f:
            ts_all = np.copy(f["ts"]) * 1e6
            acc_all = np.copy(f["acc"])
            gyr_all = np.copy(f["gyr"])
        if args.start_from_ts is not None:
            idx_start = np.where(ts_all >= args.start_from_ts)[0][0]
        else:
            idx_start = 50
        self.ts_all = ts_all[idx_start:]
        self.acc_all = acc_all[idx_start:, :]
        self.gyr_all = gyr_all[idx_start:, :]
        self.dataset_size = self.ts_all.shape[0]
        self.init_ts = self.ts_all[0]

    def load_filter(self, dataset, args):
        """
        load rotation from attitude filter and its timestamps
        """
        attitude_filter_path = osp.join(args.root_dir, dataset, "attitude.txt")
        attitudes = np.loadtxt(attitude_filter_path, delimiter=",", skiprows=3)
        self.filter_ts = attitudes[:, 0] * 1e-6
        filter_r = Rotation.from_quat(
            np.concatenate(
                [attitudes[:, 2:5], np.expand_dims(attitudes[:, 1], axis=1)], axis=1
            )
        )
        R_filter = filter_r.as_matrix()
        R_wf = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        R_filter = np.matmul(R_wf, R_filter)
        filter_r = Rotation.from_matrix(R_filter)
        self.filter_eul = filter_r.as_euler("xyz", degrees=True)

    def load_gt(self, dataset, args):
        """
        load ts, p, q, v from gt states, load ba and bg from calibration states
        """
        logging.info(
            "loading gt states from "
            + osp.join(args.root_dir, dataset, "evolving_state.txt")
        )
        gt_states = np.loadtxt(
            osp.join(args.root_dir, dataset, "evolving_state.txt"), delimiter=","
        )
        gt_calibs = np.loadtxt(
            osp.join(args.root_dir, dataset, "calib_state.txt"), delimiter=","
        )
        self.gt_ts = gt_states[:, 0] * 1e-6
        self.gt_p = gt_states[:, 5:8]
        self.gt_v = gt_states[:, 8:11]
        self.gt_rq = gt_states[:, 1:5]
        gt_r = Rotation.from_quat(
            np.concatenate(
                [self.gt_rq[:, 1:4], np.expand_dims(self.gt_rq[:, 0], axis=1)], axis=1
            )
        )
        self.gt_eul = gt_r.as_euler("xyz", degrees=True)
        self.gt_R = gt_r.as_matrix()
        self.gt_calib_ts = gt_calibs[:, 0] * 1e-6
        self.gt_ba = gt_calibs[:, 28:31]
        self.gt_bg = gt_calibs[:, 31:34]
        self.gt_accScaleInv = gt_calibs[:, 1:10].reshape((-1, 3, 3))
        self.gt_gyroScaleInv = gt_calibs[:, 10:19].reshape((-1, 3, 3))
        self.gt_gyroGSense = gt_calibs[:, 19:28].reshape((-1, 3, 3))

    def load_sim_data(self, args):
        """
        This loads simulation data from an imu.csv file containing
        perfect imu data.
        """
        logging.info("loading simulation data from " + args.sim_data_path)
        sim_data = np.loadtxt(
            args.sim_data_path,
            delimiter=",",
            usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
        )
        ts_all = sim_data[:, 0]
        gt_p = sim_data[:, 1:4]
        gt_rq = sim_data[:, 4:8]
        acc_all = sim_data[:, 8:11]
        gt_v = sim_data[:, 11:14]
        gyr_all = sim_data[:, 14:17]

        # add sim noise and bias
        if args.add_sim_imu_noise:
            wa = args.sim_sigma_na * np.random.normal(0, 1, acc_all.shape)
            wg = args.sim_sigma_ng * np.random.normal(0, 1, gyr_all.shape)
            acc_all = acc_all + wa
            gyr_all = gyr_all + wg

            sim_ba = np.array([0.3, -0.2, 0.4])
            sim_bg = np.array([0.0005, 0.002, -0.001])
            acc_all = acc_all + sim_ba
            gyr_all = gyr_all + sim_bg

        if args.start_from_ts is not None:
            idx_start = np.where(ts_all >= args.start_from_ts * 1e-6)[0][0]
        else:
            idx_start = 50

        self.ts_all = ts_all[idx_start:] * 1e6
        self.acc_all = 0.5 * (acc_all[idx_start:, :] + acc_all[idx_start - 1 : -1, :])
        self.gyr_all = 0.5 * (gyr_all[idx_start:, :] + gyr_all[idx_start - 1 : -1, :])
        # self.acc_all = acc_all[idx_start:,:]
        # self.gyr_all = gyr_all[idx_start:,:]
        self.gt_ts = ts_all[idx_start - 1 :]
        self.gt_p = gt_p[idx_start - 1 :, :]
        self.gt_v = gt_v[idx_start - 1 :, :]
        self.gt_rq = gt_rq[idx_start - 1 :, :]

        gt_r = Rotation.from_quat(self.gt_rq)
        self.gt_eul = gt_r.as_euler("xyz", degrees=True)
        self.gt_R = gt_r.as_matrix()

        self.dataset_size = self.ts_all.shape[0]
        self.init_ts = self.ts_all[0]

    def get_datai(self, idx):
        ts = self.ts_all[idx] * 1e-6  # s
        acc = self.acc_all[idx, :].reshape((3, 1))
        gyr = self.gyr_all[idx, :].reshape((3, 1))
        return ts, acc, gyr

    def get_meas_from_gt(self, ts_oldest_state, ts_end):
        """
        helper function This extracts a fake measurement from gt,
        can be used for debug to bypass the network
        """
        # obtain gt_Ri for rotating to relative frame
        idx_left = np.where(self.gt_ts < ts_oldest_state)[0][-1]
        idx_right = np.where(self.gt_ts > ts_oldest_state)[0][0]
        interp_gt_ts = self.gt_ts[idx_left : idx_right + 1]
        interp_gt_eul = self.gt_eul[idx_left : idx_right + 1, :]
        gt_euls_uw = unwrap_rpy(interp_gt_eul)
        gt_eul_uw = interp1d(interp_gt_ts, gt_euls_uw, axis=0)(ts_oldest_state)
        gt_eul = np.deg2rad(wrap_rpy(gt_eul_uw))
        ts_interp = np.array([ts_oldest_state, ts_end])
        gt_interp = interp1d(self.gt_ts, self.gt_p, axis=0)(ts_interp)
        gt_meas = gt_interp[1] - gt_interp[0]  # simulated displacement measurement
        meas_cov = np.diag(np.array([1e-2, 1e-2, 1e-2]))  # [3x3]
        # express in gravity aligned frame bty normalizing on yaw
        Ri_z = Rotation.from_euler("z", gt_eul[2]).as_matrix()
        meas = Ri_z.T.dot(gt_meas.reshape((3, 1)))
        return meas, meas_cov
