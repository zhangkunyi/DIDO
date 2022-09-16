"""
Pytorch dataloader for FB dataset
"""

import random
from abc import ABC, abstractmethod
from os import path as osp

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from numpy import random
from liegroups.numpy.so3 import SO3Matrix
from liegroups.torch.so3 import SO3Quaternion
from pyquaternion import Quaternion
from scipy import signal
import sys

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

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

from torch.nn import functional as F
import time

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

def q_integrate(gyr, ts_win,q0):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    # st_t = time.time()
    feat_gyr = gyr
    dalte_w = (feat_gyr[1:] + feat_gyr[:-1]) / 2
    # dalte_w = feat_gyr[:-1]
    dalte_w_norm = np.linalg.norm(dalte_w, ord=2, axis=1, keepdims=True)
    dalte_intint = dalte_w_norm * np.expand_dims(ts_win[0:], 1) / 2 # 因为我输入的是dt，所以这边原本的ts_win[1:]改成了ts_win[0:]
    w_point = dalte_w / dalte_w_norm
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

def jifen_q(gyr, ts_win):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    # gyr = gyr.permute(0,2,1)
    # ts_win = ts_win.permute(0,2,1)
    feat_gyr = gyr
    dalte_w = (feat_gyr[:,1:,:] + feat_gyr[:,:-1,:]) / 2
    # dalte_w = feat_gyr[:-1]
    dalte_w_norm = torch.norm(dalte_w, p=2, dim=2, keepdim=True)
    dalte_intint = dalte_w_norm * ts_win[:,1:] / 2
    w_point = dalte_w / dalte_w_norm
    dalte_q_w = torch.cos(dalte_intint)
    dalte_q_xyz = w_point * torch.sin(dalte_intint)
    dalte_q_wxyz = torch.cat((dalte_q_w, dalte_q_xyz), dim=2)

    q_inte = dalte_q_wxyz[:,0,:]

    q_win = q_inte.clone()
    for i in range(1, dalte_q_wxyz.shape[1]):
        q_inte = qmul(q_inte,dalte_q_wxyz[:,i,:])
        q_inte = F.normalize(q_inte)
        q_inte = torch.where((q_inte[:, 0] > 0).reshape(-1, 1), q_inte, -q_inte)
        q_win = torch.cat((q_win,q_inte),dim=0)
    q_inte = F.normalize(q_inte)
    q_inte = torch.where((q_inte[:,0]>0).reshape(-1,1),q_inte,-q_inte)

    return q_inte,q_win


class CompiledSequence(ABC):
    """
    An abstract interface for compiled sequence.
    """

    def __init__(self, **kwargs):
        super(CompiledSequence, self).__init__()

    @abstractmethod
    def load(self, path,args):
        pass

    @abstractmethod
    def get_feature(self):
        pass

    @abstractmethod
    def get_aux(self):
        pass

    @abstractmethod
    def get_trans_t(self):
        pass

    @abstractmethod
    def get_r(self):
        pass

    @abstractmethod
    def get_q_inte(self):
        pass

class FbSequence(CompiledSequence):
    def __init__(self, data_path, args, data_window_config,**kwargs):
        super().__init__(**kwargs)
        (
            self.ts,
            self.features,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
            self.gt_vel,

        ) = (None, None, None, None, None, None)
        self.target_dim = args.output_dim  # 网络的输出维度 3
        self.imu_freq = args.imu_freq
        self.imu_base_freq = args.imu_base_freq
        self.interval = int(data_window_config["window_size"])
        self.mode = kwargs.get("mode", "train")

        if data_path is not None:
            self.load(data_path, args)

    def load(self, data_path, args):

        with h5py.File(osp.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"]) # timestamp
            gt_p = np.copy(f["gt_p"])  # position in world frame
            gt_v = np.copy(f["gt_v"])  # velocity in world frame
            gt_q = np.copy(f["gt_q"])  # quaternion of body frame in world frame
            gyr = np.copy(f["gyr"])

        with open(osp.join("../../Rotation_ekf/output/net_acc/" + data_path[14:] + "_acc.txt"), encoding='utf-8') as f:
            print("load rotation stage acc ")
            acc = np.loadtxt(f, delimiter=",")

        # subsample from IMU base rate: 为了获得神经网络效果好的数据
        subsample_factor = int(np.around(self.imu_base_freq / self.imu_freq))
        ts = ts[::subsample_factor]
        gt_q = gt_q[::subsample_factor, :]  # 根据subsample_factor的间隔下采样
        gt_p = gt_p[::subsample_factor, :]
        gt_v = gt_v[::subsample_factor, :]
        gyr = gyr[::subsample_factor, :]
        acc = acc[::subsample_factor, :]

        if self.mode in ['val','test']:
            with open(osp.join("../../Rotation_ekf/output/ekf_q/" + data_path[14:] + "_ekf_q.txt"), encoding='utf-8') as f:
                print("load rotation stage q for val and test")
                q_inte = np.loadtxt(f, delimiter=",")
        else:
            q_inte = gt_q

        ori_R_gt = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]])  # 包含由輸入四元數表示的旋轉的對象
        ori_R = ori_R_gt

        self.trans_t = ts[self.interval:] - ts[: -self.interval]
        self.ts = ts  # ts of the beginning of each window
        self.features = np.concatenate([gyr, acc], axis=1)  # shape = Nx（3+3）
        self.orientations = ori_R.as_quat()
        self.gt_pos = gt_p
        self.gt_vel = gt_v
        self.gt_ori = ori_R_gt.as_quat()
        self.ori_r = ori_R.as_matrix()
        self.q_inte = q_inte

    def get_feature(self):
        return self.features

    def get_aux(self):
        return np.concatenate(
            [self.ts[:, None], self.orientations, self.gt_pos, self.gt_vel, self.gt_ori], axis=1
        )

    def get_trans_t(self):
        return self.trans_t

    def get_r(self):
        return self.ori_r

    def get_q_inte(self):
        return self.q_inte


class FbSequenceDataset(Dataset):
    def __init__(self, root_dir, data_list, args, data_window_config, **kwargs):
        super(FbSequenceDataset, self).__init__()

        self.window_size = data_window_config["window_size"]  # 继承得到的data_size
        self.past_data_size = data_window_config["past_data_size"]
        self.future_data_size = data_window_config["future_data_size"]
        self.step_size = data_window_config["step_size"]

        self.mode = kwargs.get("mode", "train")
        self.shuffle, self.transform = False, False
        if self.mode == "train":
            self.shuffle = False
        elif self.mode == "val":
            self.shuffle = False
        elif self.mode == "test":
            self.shuffle = False
        elif self.mode == "eval":
            self.shuffle = False

        self.index_map = []
        self.ts, self.gt_pos, self.gt_vel, self.gt_ori, self.ori_r, = [], [], [], [], []
        self.features = []
        self.glob_gyr, self.glob_acc = [], []
        self.trans_t = []
        self.q_inte = []

        for i in range(len(data_list)):
            seq = FbSequence(
                osp.join(root_dir, data_list[i]), args, data_window_config, **kwargs
            )
            feat, aux = seq.get_feature(), seq.get_aux()
            trans_t = seq.get_trans_t()
            ori_r = seq.get_r()
            q_inte = seq.get_q_inte()

            self.features.append(feat)
            self.ts.append(aux[:, 0])
            self.gt_pos.append(aux[:, 5:8])
            self.gt_vel.append(aux[:, 8:11])
            self.gt_ori.append(aux[:, 11:15])
            self.trans_t.append(trans_t)
            self.ori_r.append(ori_r)
            self.q_inte.append(q_inte)

            self.index_map += [
                [i, j]
                for j in range(
                    0 + self.past_data_size,
                    self.trans_t[i].shape[0] - self.future_data_size,
                    self.step_size,
                )
            ]
        if self.shuffle:
            random.shuffle(self.index_map)  # 将序列中的元素随机打乱，用于训练（train mode）

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]  # frame_id是window开始端的id

        feat = self.features[seq_id][
            frame_id
            - self.past_data_size :  frame_id
            + self.window_size
            + self.future_data_size
        ]

        gt_v = self.gt_vel[seq_id][frame_id]
        gt_p = self.gt_pos[seq_id][frame_id]
        try:
            trans_t = self.trans_t[seq_id][frame_id]
        except:
            print('a')
        gt_v_all = self.gt_vel[seq_id][
            frame_id
            - self.past_data_size :  frame_id
            + self.window_size
            + self.future_data_size
        ]
        gt_p_all = self.gt_pos[seq_id][
            frame_id
            - self.past_data_size :  frame_id
            + self.window_size
            + self.future_data_size
        ]

        # Calculate the imu velocity integral within a time window
        ts_inter = self.ts[seq_id][
               frame_id:  frame_id
                           + self.window_size
                           + self.future_data_size
               ]
        ts_inter_old = np.append(ts_inter[0], ts_inter[:-1])
        dt = ts_inter - ts_inter_old   # s

        time = self.ts[seq_id][
            frame_id
            - self.past_data_size :  frame_id
            + self.window_size
            + self.future_data_size
        ]

        ori_r = self.ori_r[seq_id][
            frame_id
            - self.past_data_size :  frame_id
            + self.window_size
            + self.future_data_size
        ]

        q_inte = self.q_inte[seq_id][
            frame_id
            - self.past_data_size :  frame_id
            + self.window_size
            + self.future_data_size
        ]

        return feat.astype(np.float32).T, gt_v.astype(np.float32),trans_t.astype(np.float32),dt.astype(np.float32),\
               gt_v_all.astype(np.float32),gt_p.astype(np.float32),gt_p_all.astype(np.float32),time.astype(np.float64),\
               ori_r.astype(np.float32),q_inte.astype(np.float32),seq_id, frame_id

    def __len__(self):
        return len(self.index_map)
