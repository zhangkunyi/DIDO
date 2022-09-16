"""
Pytorch dataloader for FB dataset
"""

import random
from abc import ABC, abstractmethod
from os import path as osp

import h5py, time
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn import functional as F


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

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
    terms = torch.bmm(r.contiguous().view(-1, 4, 1), q.contiguous().view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape).to(q.device)

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

class FbSequence(CompiledSequence):
    def __init__(self, data_path, args, data_window_config, **kwargs):
        super().__init__(**kwargs)
        (
            self.ts,
            self.feat_body,
            self.gamma_q_win,
        ) = (None, None, None)
        self.target_dim = args.output_dim
        self.imu_freq = args.imu_freq
        self.imu_base_freq = args.imu_base_freq
        self.interval = data_window_config["window_size"]
        self.mode = kwargs.get("mode", "train")

        if data_path is not None:
            self.load(data_path, args)

    def load(self, data_path, args):
        with h5py.File(osp.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])  # timestamp
            gt_p = np.copy(f["gt_p"])  # position in world frame
            gt_v = np.copy(f["gt_v"])  # velocity in world frame
            gt_q = np.copy(f["gt_q"])  # quaternion of body frame in world frame
            gyr = np.copy(f["gyr"])  # unbiased gyr
            acc = np.copy(f["acc"])  # unbiased acc

        # subsample from IMU base rate:
        subsample_factor = int(np.around(self.imu_base_freq / self.imu_freq))
        ts = ts[::subsample_factor]
        gt_q = gt_q[::subsample_factor, :]
        gyr_body = gyr[::subsample_factor, :]
        acc_body = acc[::subsample_factor, :]

        q0_norm = gt_q[: -self.interval] / np.linalg.norm(gt_q[: -self.interval], axis=1, keepdims=True)
        q1_norm = gt_q[self.interval:] / np.linalg.norm(gt_q[self.interval:], axis=1, keepdims=True)
        q_rela_torch = F.normalize(
            qmul(torch.tensor(q0_norm) * torch.tensor([1, -1, -1, -1]), torch.tensor(q1_norm)))
        gamma_q_win = q_rela_torch.numpy()
        gamma_q_win[gamma_q_win[:, 0] < 0] = (-1) * gamma_q_win[gamma_q_win[:, 0] < 0]
        gamma_q_win = np.array(gamma_q_win)  # increment of rotation
        self.gamma_q_win = gamma_q_win
        self.ts = ts  # ts of the beginning of each window
        self.feat_body = np.concatenate([gyr_body, acc_body], axis=1)

    def get_feature(self):
        return self.feat_body

    def get_target(self):
        return self.gamma_q_win

    def get_aux(self):
        return np.concatenate(
            [self.ts[:, None]], axis=1
        )

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
            self.shuffle = False
            self.transform = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False
        elif self.mode == "eval":
            self.shuffle = False

        self.index_map = []
        self.ts = []
        self.feat_body = []
        self.gamma_q_win = []
        self.arch = args.arch
        # each dataset is a list
        for i in range(len(data_list)):
            seq = FbSequence(
                osp.join(root_dir, data_list[i]), args, data_window_config, **kwargs
            )
            gamma_q_win = seq.get_target()
            aux = seq.get_aux()
            feat_body = seq.get_feature()
            self.feat_body.append(feat_body)
            self.gamma_q_win.append(gamma_q_win)
            self.ts.append(aux[:, 0])

            self.index_map += [
                [i, j]
                for j in range(
                    0 + self.past_data_size,
                    self.gamma_q_win[i].shape[0] - self.future_data_size,
                    self.step_size,
                )
            ]

        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        # get feature to train
        feat = self.feat_body[seq_id][
            frame_id
            - self.past_data_size: frame_id
            + self.window_size
            + self.future_data_size
        ]

        gamma_q_win = self.gamma_q_win[seq_id][frame_id]

        # Calculate the imu velocity integral within a time window
        ts_inter = self.ts[seq_id][
                   frame_id
                   - self.past_data_size: frame_id
                   + self.window_size
                   + self.future_data_size
               ]

        d_t_tmp = np.diff(ts_inter)
        d_t = np.append(d_t_tmp[0], d_t_tmp)

        feat_target = {
            "feat": feat.astype(np.float32).T,
            "gamma_q_win": gamma_q_win.astype(np.float32),
            "d_t": d_t.astype(np.float32),
        }
        return feat_target, seq_id, frame_id

    def __len__(self):
        return len(self.index_map)
