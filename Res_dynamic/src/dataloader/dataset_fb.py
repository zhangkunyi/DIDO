"""
Pytorch dataloader for FB dataset
"""

import random
from abc import ABC, abstractmethod
from os import path as osp

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from pyquaternion import Quaternion


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_target(self):
        pass

    @abstractmethod
    def get_r(self):
        pass

    @abstractmethod
    def get_gt_v(self):
        pass

def q_integrate(gyr, ts_win,q0):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    # st_t = time.time()
    feat_gyr = gyr
    dalte_w = (feat_gyr[1:] + feat_gyr[:-1]) / 2
    # dalte_w = feat_gyr[:-1]
    dalte_gyr_norm = np.linalg.norm(dalte_w, ord=2, axis=1, keepdims=True)
    dalte_intint = dalte_gyr_norm * np.expand_dims(ts_win[0:], 1) / 2 # 因为我输入的是dt，所以这边原本的ts_win[1:]改成了ts_win[0:]
    w_point = dalte_w / dalte_gyr_norm
    dalte_q_w = np.cos(dalte_intint)
    dalte_q_xyz = w_point * np.sin(dalte_intint)
    dalte_q_wxyz = np.concatenate((dalte_q_w, dalte_q_xyz), axis=1)

    # dalte_q_1 = Quaternion(np.array([1, 0, 0, 0]))
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

        # print(iii)
    dalte_q_x_xnorm = dalte_q_x.normalised
    dalte_q_diff = dalte_q_x_xnorm.q
    dalte_q_diff = np.array(dalte_q_diff)
    dalte_q_win = np.array(dalte_q_win)

    # end_t = time.time()
    # print("计算时间：", end_t - st_t)
    return dalte_q_diff, dalte_q_win

class FbSequence(CompiledSequence):
    def __init__(self, data_path, args, data_window_config, **kwargs):
        super().__init__(**kwargs)
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.target_dim = args.output_dim
        self.imu_freq = args.imu_freq
        self.imu_base_freq = args.imu_base_freq
        self.interval = data_window_config["window_size"]
        self.mode = kwargs.get("mode", "train")

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):

        with h5py.File(osp.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            gt_q_imu = np.copy(f["gt_q"])
            gt_v_imu = np.copy(f["gt_v"])
            acc_imu = np.copy(f["acc"])
            gt_acc = np.copy(f["gt_acc"])
            gt_gyr = np.copy(f["gt_gyr"])
            gt_alpha = np.copy(f["gt_alpha"])
            dynamic_param = np.copy(f["dynamic_params"])
            rpm = np.copy(f["meas_rpm"])

        with open(osp.join("../../Rotation_ekf/output/net_gyr/"+data_path[14:] + "_gyr.txt"), encoding='utf-8') as f:
            net_gyr_imu = np.loadtxt(f, delimiter=",")
        if self.mode in ['val','test']:
            with open(osp.join("../../Rotation_ekf/output/ekf_q/" + data_path[14:] + "_ekf_q.txt"), encoding='utf-8') as f:
                print("load rotation stage q for val and test")
                gt_q_imu = np.loadtxt(f, delimiter=",")


        kf = dynamic_param[0:1]
        D = dynamic_param[1:4]
        trans_ex = dynamic_param[4:7]  # 在imu系下看，imu到rigid的平移
        pitch_ex = dynamic_param[7]
        roll_ex = dynamic_param[8]
        rot_dcm = Rotation.from_euler("xyz", np.array([roll_ex, pitch_ex, 0]),degrees=True).as_matrix().T  # rigid2imu

        # transfer to rigid frame
        dcm_imu = Rotation.from_quat(gt_q_imu[:, [1, 2, 3, 0]]).as_matrix()
        gyr_gra = np.einsum('tip,tp->ti', dcm_imu, gt_gyr)
        trans_gra = np.matmul(dcm_imu, trans_ex)
        alpha_gra = np.einsum('tip,tp->ti', dcm_imu, gt_alpha)
        gt_v_rigid = gt_v_imu + np.cross(gyr_gra, trans_gra)

        dcm_rigid = np.matmul(dcm_imu, rot_dcm)
        gt_q_rigid = Rotation.from_matrix(dcm_rigid).as_quat()
        gt_q_rigid = gt_q_rigid[:,[3, 0, 1, 2]]  #
        net_gyr_rigid = np.einsum('ip,tp->ti', rot_dcm.T, net_gyr_imu)

        gt_acc_rigid_body = gt_acc + np.cross(gyr_gra, np.cross(gyr_gra, trans_gra)) + np.cross(alpha_gra,trans_gra)
        gt_acc_rigid_body = np.einsum("tpi,tp->ti", dcm_rigid, gt_acc_rigid_body)

        dt = np.expand_dims(np.append(np.diff(ts), np.diff(ts)[-1]),axis=1)

        # subsample from IMU base rate:
        subsample_factor = int(np.around(self.imu_base_freq / self.imu_freq))
        ts = ts[::subsample_factor]
        gt_q = gt_q_rigid[::subsample_factor, :]
        gyro = net_gyr_rigid[::subsample_factor, :]
        acc = acc_imu[::subsample_factor, :]
        rpm = rpm[::subsample_factor, :]
        gt_acc_rigid_body = gt_acc_rigid_body[::subsample_factor, :]
        gt_v_rigid = gt_v_rigid[::subsample_factor, :]

        # rotation in the world frame in quaternions
        ori_R_gt = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]])
        ori_R = ori_R_gt

        self.ts = ts  # ts of the beginning of each window
        self.features = np.concatenate([gyro, acc], axis=1)
        # self.features = motor
        self.orientations = ori_R.as_quat()
        self.gt_ori = ori_R_gt.as_quat()
        self.ori_r = ori_R.as_matrix()
        self.targets = gt_acc_rigid_body[:, : self.target_dim]
        self.gt_v = gt_v_rigid
        self.gt_q = gt_q
        self.kf = kf
        self.D = D
        self.rpm = rpm

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_r(self):
        return self.ori_r

    def get_gt_v(self):
        return self.gt_v

    def get_gt_q(self):
        return self.gt_q

    def get_kf(self):
        return self.kf

    def get_D(self):
        return self.D

    def get_rpm(self):
        return self.rpm

class FbSequenceDataset(Dataset):
    def __init__(self, root_dir, data_list, args, data_window_config, **kwargs):
        super(FbSequenceDataset, self).__init__()

        self.window_size = data_window_config["window_size"]
        self.past_data_size = data_window_config["past_data_size"]
        self.future_data_size = data_window_config["future_data_size"]
        self.step_size = data_window_config["step_size"]

        self.mode = kwargs.get("mode", "train")
        self.shuffle, self.transform = False, False
        if self.mode == "train":
            self.shuffle = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False
        elif self.mode == "eval":
            self.shuffle = False

        self.index_map = []
        self.ts, self.orientations, self.ori_r, self.gt_v = [], [], [], []
        self.features, self.targets = [], []
        self.no_yaw_q = []
        self.gt_q = []
        self.rpm = []
        self.kf = []
        self.D = []

        for i in range(len(data_list)):
            seq = FbSequence(
                osp.join(root_dir, data_list[i]), args, data_window_config, **kwargs
            )
            kf = seq.get_kf()
            D = seq.get_D()
            feat, targ, ori_r, gt_v, gt_q, rpm = seq.get_feature(), seq.get_target(), seq.get_r(), seq.get_gt_v(), seq.get_gt_q(), seq.get_rpm()
            self.features.append(feat)
            self.targets.append(targ)
            self.ori_r.append(ori_r)
            self.gt_q.append(gt_q)
            self.kf.append(kf)
            self.D.append(D)
            self.rpm.append(rpm)
            self.index_map += [
                [i, j]
                for j in range(
                    0 + self.past_data_size,
                    self.targets[i].shape[0] - self.future_data_size,
                    self.step_size,
                )
            ]

            self.gt_v.append(gt_v)

        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        # in the world frame
        feat = self.features[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]
        targ = self.targets[seq_id][frame_id]  # the beginning of the sequence

        ori_r = self.ori_r[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]

        gt_v = self.gt_v[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]
        gt_q = self.gt_q[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]

        rpm = self.rpm[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]

        kf = self.kf[seq_id]
        D = self.D[seq_id]


        return feat.astype(np.float32).T, targ.astype(np.float32), ori_r.astype(np.float32), gt_v.astype(np.float32), gt_q.astype(np.float32), rpm.astype(np.float32), kf.astype(np.float32), D.astype(np.float32), seq_id, frame_id
    def __len__(self):
        return len(self.index_map)
