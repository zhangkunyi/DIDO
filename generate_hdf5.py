import numpy as np
import h5py
from os import path as osp
import os
import tqdm

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list

if __name__ == '__main__':
    root_dir = 'only_hdf5_gra_aligned_adjust_q/'
    out_dir = 'dataset'
    test_path = 'only_hdf5_gra_aligned_adjust_q/gen_list.txt'
    data_path_s = get_datalist(test_path)
    for data_path in tqdm.tqdm(data_path_s):
        with h5py.File(root_dir + '/' + data_path + '/data.hdf5', "r") as f:
            ts = np.copy(f["ts"])  # timestamp
            gt_p = np.copy(f["vio_p"])  # position in world frame
            gt_v = np.copy(f["vio_v"])  # velocity in world frame
            gt_q = np.copy(f["vio_q"])  # quaternion of body frame in world frame
            gyr = np.copy(f["gyr_raw"])  # unbiased gyr
            acc = np.copy(f["acc_raw"])
            spline_data = np.copy(f["spline_fit"])

        with open(osp.join(root_dir + '/' + data_path +  "/meas_rpm.txt"),
                  encoding='utf-8') as f:
            meas_rpm = np.loadtxt(f, delimiter=",") / 1e4

        with open(osp.join(root_dir + '/' + data_path +  "/t_d_tra_pr_gt_v_by_a.txt"),
                  encoding='utf-8') as f:
            dynamic_params = np.loadtxt(f, delimiter=",")

        gt_acc = spline_data[:, 6:9]
        gt_gyr = spline_data[:, 13:16]
        gt_alpha = spline_data[:, 16:19]

        if not osp.isdir(out_dir + '/' + data_path):
            os.makedirs(out_dir + '/' + data_path)

        with h5py.File(osp.join(out_dir + '/' + data_path, "data.hdf5"), "w") as f:
            f.create_dataset("ts", data=ts)
            f.create_dataset("gt_acc", data=gt_acc)
            f.create_dataset("gt_gyr", data=gt_gyr)
            f.create_dataset("gt_alpha", data=gt_alpha)
            f.create_dataset("gt_p", data=gt_p)
            f.create_dataset("gt_v", data=gt_v)
            f.create_dataset("gt_q", data=gt_q)
            f.create_dataset("acc", data=acc)
            f.create_dataset("gyr", data=gyr)
            f.create_dataset("meas_rpm", data=meas_rpm)
            f.create_dataset("dynamic_params", data=dynamic_params)
