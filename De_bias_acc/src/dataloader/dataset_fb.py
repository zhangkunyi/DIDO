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
    def get_aux(self):
        pass

    @abstractmethod
    def get_r(self):
        pass

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
            gt_q = np.copy(f["gt_q"])
            gt_v = np.copy(f["gt_v"])  # velocity in world frame
            gt_p = np.copy(f["gt_p"])
            gyro = np.copy(f["gyr"])
            acc = np.copy(f["acc"])

        # subsample from IMU base rate:
        subsample_factor = int(np.around(self.imu_base_freq / self.imu_freq))
        ts = ts[::subsample_factor]
        gt_q = gt_q[::subsample_factor, :]
        gt_p = gt_p[::subsample_factor, :]
        gyro = gyro[::subsample_factor, :]
        acc = acc[::subsample_factor, :]

        dt = np.expand_dims(np.diff(ts), 1)

        # ground truth displacement
        gt_dv = gt_v[self.interval :] - gt_v[: -self.interval]

        ori_R_gt = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]])

        ori_R = ori_R_gt

        self.ts = ts  # ts of the beginning of each window
        self.features = np.concatenate([gyro, acc], axis=1)  ## jcx
        self.orientations = ori_R.as_quat()
        self.gt_pos = gt_p
        self.gt_ori = ori_R_gt.as_quat()
        self.ori_r = ori_R.as_matrix()
        self.targets = gt_dv[:, : self.target_dim]
        self.gt_vel = gt_v

    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate(
            [self.ts[:, None], self.orientations, self.gt_pos, self.gt_ori], axis=1
        )
    def get_r(self):
        return self.ori_r

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
            self.transform = False
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False
        elif self.mode == "eval":
            self.shuffle = False

        self.index_map = []
        self.ts, self.orientations, self.gt_pos, self.gt_ori, self.ori_r = [], [], [], [],[]
        self.features, self.targets = [], []
        self.no_yaw_q = []
        self.gt_q = []
        self.gt_vel = []
        for i in range(len(data_list)):
            seq = FbSequence(
                osp.join(root_dir, data_list[i]), args, data_window_config, **kwargs
            )
            feat, targ, aux ,ori_r = seq.get_feature(), seq.get_target(), seq.get_aux(), seq.get_r()
            self.features.append(feat)
            self.targets.append(targ)
            self.ts.append(aux[:, 0])
            self.orientations.append(aux[:, 1:5])
            self.gt_pos.append(aux[:, 5:8])
            self.gt_ori.append(aux[:, 8:12])
            self.ori_r.append(ori_r)
            self.gt_vel.append(seq.gt_vel)
            self.index_map += [
                [i, j]
                for j in range(
                    0 + self.past_data_size,
                    self.targets[i].shape[0] - self.future_data_size,
                    self.step_size,
                )
            ]

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

        ts_inter = self.ts[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]
        ts_inter_old = np.append(ts_inter[0], ts_inter[:-1])
        dt = ts_inter - ts_inter_old
        ori_r = self.ori_r[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]

        return feat.astype(np.float32).T, targ.astype(np.float32), dt, ori_r.astype(np.float32) ,seq_id, frame_id

    def __len__(self):
        return len(self.index_map)
