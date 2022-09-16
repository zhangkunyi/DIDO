"""
IMU network training/testing/evaluation for displacement and covariance
Input: Nx6 IMU data
Output: 3x1 displacement, 3x1 covariance parameters
"""

import network
from utils.argparse_utils import add_bool_arg
import os
import numpy as np
import torch

if __name__ == "__main__":

    myseed = 1
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_list", type=str, default="../../dataset/train.txt")
    parser.add_argument("--val_list", type=str, default="../../dataset/val.txt")
    parser.add_argument("--test_list", type=str, default="../../dataset/test.txt")
    parser.add_argument(
        "--root_dir", type=str, default="../../dataset", help="Path to data directory"
    )

    parser.add_argument("--out_dir", type=str, default="../train_output")
    parser.add_argument("--model_path", type=str, default="/home/jiangcx/桌面/TLIO/TLIO_raw_a/gra_aligned_adjust_q_data/net_d/cov/no_q/checkpoint_1113.pt")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--out_name", type=str, default=None)

    # ------------------ architecture and training -----------------
    parser.add_argument("--lr", type=float, default=1e-04)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10000, help="max num epochs")
    parser.add_argument("--arch", type=str, default="resnet")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--input_dim", type=int, default=10)
    parser.add_argument("--output_dim", type=int, default=3)

    # ------------------ commons -----------------
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "eval"]
    )
    parser.add_argument(
        "--imu_freq", type=float, default=400.0, help="imu_base_freq is a multiple"
    )
    parser.add_argument("--imu_base_freq", type=float, default=400.0)

    # ----- window size and inference freq -----
    parser.add_argument("--past_time", type=float, default=0.05-0.0025)  # s
    parser.add_argument("--window_time", type=float, default=0.0025)  # s
    parser.add_argument("--future_time", type=float, default=0.0)  # s

    # ----- for sampling in training / stepping in testing -----
    parser.add_argument("--sample_freq", type=float, default=400.0)  # hz

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
