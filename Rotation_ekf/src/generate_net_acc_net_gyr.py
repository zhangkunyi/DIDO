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

def output_debias_net(args):

    network_acc_path = args.network_acc_path
    network_gyr_path = args.network_gyr_path
    device = args.device
    win_size = args.win_size
    batch_size = args.batch_size
    imu_freq = args.imu_freq
    net_config = {"in_dim": int((win_size * imu_freq) // 4)}

    network_acc = get_model("resnet", net_config, int(3), int(3)).to(device)
    checkpoint = torch.load(network_acc_path, map_location="cuda")
    network_acc.load_state_dict(checkpoint["model_state_dict"])
    network_acc.eval()

    network_gyr = get_model("resnet", net_config, int(3), int(3)).to(device)
    checkpoint = torch.load(network_gyr_path, map_location="cuda")
    network_gyr.load_state_dict(checkpoint["model_state_dict"])
    network_gyr.eval()

    root_dir = args.root_dir
    out_dir = args.out_dir
    test_path = args.test_list
    data_path_s = get_datalist(test_path)
    for data_path in tqdm.tqdm(data_path_s):
        with h5py.File(root_dir + '/' + data_path + '/data.hdf5', "r") as f:
            ts = np.copy(f["ts"])  # timestamp
            gt_p = np.copy(f["gt_p"])  # position in world frame
            gt_v = np.copy(f["gt_v"])  # velocity in world frame
            gt_q = np.copy(f["gt_q"])  # quaternion of body frame in world frame
            gyr = np.copy(f["gyr"])  # unbiased gyr
            acc = np.copy(f["acc"])
            gt_acc = np.copy(f["gt_acc"])
            gt_gyr = np.copy(f["gt_gyr"])

        gt_R = Rotation.from_quat(gt_q[:, [1, 2, 3, 0]]).as_matrix()
        gt_acc = np.einsum("tpi,tp->ti", gt_R, gt_acc + np.array([0, 0, 9.8]))  # 特别注意 jcx 而且是旋转的转置

        dt = np.diff(ts)
        dt = np.append(dt[0], dt).reshape(-1, 1)

        acc = acc.astype(np.float32)
        batch_acc = []
        batch_gyr = []

        start_step = 0

        for i in range(start_step, int(acc.shape[0] - (imu_freq * win_size))):
            batch_acc.append(acc[i:i + int(imu_freq * win_size)])
            batch_gyr.append(gyr[i:i + int(imu_freq * win_size)])

        batch_acc = np.array(batch_acc).astype(np.float32)
        batch_acc = torch.tensor(batch_acc).to(device)
        batch_acc = batch_acc.permute(0, 2, 1)

        batch_gyr = np.array(batch_gyr).astype(np.float32)
        batch_gyr = torch.tensor(batch_gyr).to(device)
        batch_gyr = batch_gyr.permute(0, 2, 1)

        ba_all = []
        bg_all = []
        for i in range(int(batch_acc.shape[0] // batch_size)):
            ba = network_acc(batch_acc[i * batch_size: (i + 1) * batch_size])
            ba_all.append(torch_to_numpy(ba))

            bg = network_gyr(batch_gyr[i * batch_size: (i + 1) * batch_size])
            bg_all.append(torch_to_numpy(bg))

        ba = network_acc(batch_acc[(int(batch_acc.shape[0] // batch_size)) * batch_size:])
        ba_all.append(torch_to_numpy(ba))
        ba_all = np.concatenate(ba_all, axis=0)

        bg = network_gyr(batch_gyr[(int(batch_acc.shape[0] // batch_size)) * batch_size:])
        bg_all.append(torch_to_numpy(bg))
        bg_all = np.concatenate(bg_all, axis=0)

        pred_a = acc[start_step + int(win_size * imu_freq):] + ba_all
        pred_w = gyr[start_step + int(win_size * imu_freq):] + bg_all

        pred_gyr_all = np.concatenate((gt_gyr[:int(win_size * imu_freq)], pred_w), axis=0)
        pred_acc_all = np.concatenate((gt_acc[:int(win_size * imu_freq)], pred_a), axis=0)

        if args.save_result:
            file_names = ['net_gyr','net_acc']
            for file_name in file_names:
                if not osp.isdir( out_dir + '/'+file_name+'/'):
                    os.makedirs(out_dir + '/'+file_name+'/')
                    print('create '+ out_dir + '/'+file_name+'/')
            np.savetxt(out_dir+'/net_gyr/' + data_path + '_gyr.txt', pred_gyr_all, delimiter=',')
            np.savetxt(out_dir+'/net_acc/' + data_path + '_acc.txt', pred_acc_all, delimiter=',')

    print('a')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_acc_path", type=str, default= "../../De_bias_acc/train_output/checkpoints/checkpoint_495.pt")
    parser.add_argument("--network_gyr_path", type=str, default="../../De_bias_gyr/train_output/checkpoints/checkpoint_199.pt")
    parser.add_argument("--win_size", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1256)
    parser.add_argument("--imu_freq", type=int, default=400)
    parser.add_argument("--save_result", type=bool, default=True)
    parser.add_argument("--show_figure", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_list", type=str, default="../../dataset/gen_list.txt")
    parser.add_argument("--out_dir", type=str, default="../output")
    parser.add_argument(
        "--root_dir", type=str, default="../../dataset", help="Path to data directory")

    args = parser.parse_args()

    output_debias_net(args)
