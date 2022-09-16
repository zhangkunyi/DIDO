"""
IMU network training/testing/evaluation for displacement and covariance
Input: Nx6 IMU data
Output: 3x1 displacement, 3x1 covariance parameters
"""

import network
from utils.argparse_utils import add_bool_arg
import os
import torch
import numpy as np

if __name__ == "__main__":

    myseed = 1
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    import argparse

    base_path = os.getcwd()
    parser = argparse.ArgumentParser()

    # ------------------ directories -----------------
    # todo 用于修改的主要参数
    arch_name = 'resnet'
    past_time = 0.0
    window_time = 1.0
    out_hz = 20
    data_name = "only_hdf5_gra_aligned"
    train_mode = "train"
    path_formant = "/../../"  # for pycharm

    root_path = "../../dataset/"
    parser.add_argument("--train_list", type=str,
                        default=os.path.join(root_path + "train.txt"))
    parser.add_argument("--val_list", type=str, default=os.path.join(root_path + "val.txt"))
    parser.add_argument("--test_list", type=str,
                        default=os.path.join(root_path + "test.txt"))
    parser.add_argument(
        "--root_dir", type=str, default=root_path, help="Path to data directory")
    # todo 训练模型输出文件夹
    parser.add_argument("--out_dir", type=str, default="../train_output")
    parser.add_argument("--model_path", type=str, default="../train_output/checkpoints/checkpoint_30.pt")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)

    # ------------------ architecture and training -----------------
    parser.add_argument("--lr", type=float, default=1e-04)  #
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1000000, help="max num epochs")
    # todo 训练模型
    parser.add_argument("--arch", type=str, default=arch_name)
    parser.add_argument("--gt_q_all_size", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=3)

    # ------------------ commons -----------------
    # todo 训练模式
    parser.add_argument(
        "--mode", type=str, default=train_mode, choices=["train", "test", "eval"]
    )
    parser.add_argument(
        "--imu_freq", type=float, default=400.0, help="imu_base_freq is a multiple"  #
    )
    parser.add_argument("--imu_base_freq", type=float, default=400.0)

    # ----- window size and inference freq -----
    # todo 时间窗口大小
    parser.add_argument("--past_time", type=float, default=past_time)  # s
    parser.add_argument("--window_time", type=list, default=window_time)  # s
    parser.add_argument("--future_time", type=float, default=0.0)  # s

    # ----- for sampling in training / stepping in testing -----
    # todo 预测结果输出帧率
    parser.add_argument("--sample_freq", type=float, default=out_hz)  # hz

    # ----- plotting and evaluation -----
    add_bool_arg(parser, "save_plot", default=True)
    parser.add_argument("--rpe_window", type=float, default="2.0")  # s

    args = parser.parse_args()

    ###########################################################
    # Main
    ###########################################################
    if args.mode == "train":
        network.net_train(args)
    else:
        raise ValueError("Undefined mode")
