"""
This file includes the main libraries in the network testing module
"""

import json
import os
from os import path as osp

import matplotlib.pyplot as plt
import torch
from dataloader.dataset_fb import FbSequenceDataset
from network.losses import get_loss
from network.model_factory import get_model
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from utils.logging import logging
from utils.math_utils import *
from .train import get_inference

def vel_integrate(args, dataset, preds):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    dp_t = args.window_time
    pred_vels = preds / dp_t

    ind = np.array([i[1] for i in dataset.index_map], dtype=np.int)
    delta_int = int(
        args.window_time * args.imu_freq / 2.0
    )  # velocity as the middle of the segment
    if not (args.window_time * args.imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    ind_intg = ind + delta_int  # the indices of doing integral

    ts = dataset.ts[0]
    dts = np.mean(ts[ind_intg[1:]] - ts[ind_intg[:-1]])
    vel_intg = np.zeros([pred_vels.shape[0] + 1, args.output_dim])
    vel_intg[0] = dataset.gt_vel[0][ind_intg[0], :]
    vel_intg[1:] = np.cumsum(pred_vels[:, :] * dts, axis=0) + vel_intg[0]
    ts_intg = np.append(ts[ind_intg], ts[ind_intg[-1]] + dts)

    ts_in_range = ts[ind_intg[0] : ind_intg[-1]]  # s
    vel_pred = interp1d(ts_intg, vel_intg, axis=0)(ts_in_range)
    vel_gt = dataset.gt_vel[0][ind_intg[0] : ind_intg[-1], :]

    traj_attr_dict = {
        "ts": ts_in_range,
        "vel_pred": vel_pred,
        "vel_gt": vel_gt,
    }

    return traj_attr_dict

def compute_metrics_and_plotting(args, net_attr_dict, traj_attr_dict):
    """
    Obtain trajectory and compute metrics.
    """

    """ ------------ Trajectory metrics ----------- """
    ts = traj_attr_dict["ts"]
    vel_pred = traj_attr_dict["vel_pred"]
    vel_gt = traj_attr_dict["vel_gt"]

    # get RMSE
    rmse = np.sqrt(np.mean(np.linalg.norm(vel_pred - vel_gt, axis=1) ** 2))
    # get ATE
    diff_vel = vel_pred - vel_gt
    ate = np.mean(np.linalg.norm(diff_vel, axis=1))
    # get velocity drift
    traj_lens = np.sum(np.linalg.norm(vel_gt[1:] - vel_gt[:-1], axis=1))
    drift_vel = np.linalg.norm(vel_pred[-1, :] - vel_gt[-1, :])
    drift_ratio = drift_vel / traj_lens

    metrics = {
        "ronin": {
            "rmse": rmse,
            "ate": ate,
            "drift_vel (m/m)": drift_ratio
        }
    }

    """ ------------ Network loss metrics ----------- """
    mse_loss = np.mean(
        (net_attr_dict["targets"] - net_attr_dict["preds"]) ** 2, axis=0
    )  # 3x1
    likelihood_loss = np.mean(net_attr_dict["losses"], axis=0)  # 3x1
    avg_mse_loss = np.mean(mse_loss)
    avg_likelihood_loss = np.mean(likelihood_loss)
    metrics["ronin"]["mse_loss_x"] = float(mse_loss[0])
    metrics["ronin"]["mse_loss_y"] = float(mse_loss[1])
    metrics["ronin"]["mse_loss_z"] = float(mse_loss[2])
    metrics["ronin"]["mse_loss_avg"] = float(avg_mse_loss)
    metrics["ronin"]["likelihood_loss_x"] = float(likelihood_loss[0])
    metrics["ronin"]["likelihood_loss_y"] = float(likelihood_loss[1])
    metrics["ronin"]["likelihood_loss_z"] = float(likelihood_loss[2])
    metrics["ronin"]["likelihood_loss_avg"] = float(avg_likelihood_loss)

    """ ------------ Data for plotting ----------- """
    total_pred = net_attr_dict["preds"].shape[0]
    pred_ts = (1.0 / args.sample_freq) * np.arange(total_pred)
    plot_dict = {
        "ts": ts,
        "vel_pred": vel_pred,
        "vel_gt": vel_gt,
        "pred_ts": pred_ts,
        "preds": net_attr_dict["preds"],
        "targets": net_attr_dict["targets"],
        "rmse": rmse
    }

    return metrics, plot_dict

def plot_3d(x, y1, y2, xlb, ylbs, lgs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1[:, i], label=lgs[0])
        plt.plot(x, y2[:, i], label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig

def plot_3d_1var(x, y, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        if x is not None:
            plt.plot(x, y[:, i])
        else:
            plt.plot(y[:, i])
        plt.ylabel(ylbs[i])
        plt.grid(True)
    if xlb is not None:
        plt.xlabel(xlb)
    return fig

def plot_3d_2var_with_sigma(
    x, y1, y2, xlb, ylbs, lgs, num=None, dpi=None, figsize=None
):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1[:, i], "-b", linewidth=0.5, label=lgs[0])
        plt.plot(x, y2[:, i], "-r", linewidth=0.5, label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig

def plot_3d_err(x, y, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y[:, i], "-b", linewidth=0.5)
        plt.ylabel(ylbs[i])
        plt.grid(True)
    plt.xlabel(xlb)
    return fig

def make_plots(args, plot_dict, outdir):
    ts = plot_dict["ts"]
    vel_pred = plot_dict["vel_pred"]
    vel_gt = plot_dict["vel_gt"]
    pred_ts = plot_dict["pred_ts"]
    preds = plot_dict["preds"]
    targets = plot_dict["targets"]
    rmse = plot_dict["rmse"]

    dpi = 90
    figsize = (16, 9)

    fig1 = plt.figure(num="prediction vs gt", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    plt.plot(vel_pred[:, 0], vel_pred[:, 1])
    plt.plot(vel_gt[:, 0], vel_gt[:, 1])
    plt.axis("equal")
    plt.legend(["Predicted", "Ground truth"])
    plt.title("2D trajectory and ATE error against time")
    plt.subplot2grid((3, 2), (2, 0))
    plt.plot(np.linalg.norm(vel_pred - vel_gt, axis=1))
    plt.legend(["RMSE:{:.3f}".format(rmse)])
    for i in range(3):
        plt.subplot2grid((3, 2), (i, 1))
        plt.plot(preds[:, i])
        plt.plot(targets[:, i])
        plt.legend(["Predicted", "Ground truth"])
        plt.title("{}".format(targ_names[i]))
    plt.tight_layout()
    plt.grid(True)

    fig2 = plot_3d(
        ts,
        vel_pred,
        vel_gt,
        xlb="t(s)",
        ylbs=["x(m/s)", "y(m/s)", "z(m/s)"],
        lgs=["our", "Ground Truth"],
        num="Velocity",
        dpi=dpi,
        figsize=figsize,
    )
    fig4 = plot_3d_err(
        pred_ts,
        preds - targets,
        xlb="t(s)",
        ylbs=["x(m/s)", "y(m/s)", "z(m/s)"],
        num="Velocity errors",
        dpi=dpi,
        figsize=figsize,
    )

    fig1.savefig(osp.join(outdir, "velocity_view.png"))
    fig2.savefig(osp.join(outdir, "velocity.png"))
    fig4.savefig(osp.join(outdir, "pred-err.svg"))

    plt.close("all")

    return


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list

def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dict(
        [
            ("past_data_size", int(args.past_time * args.imu_freq)),
            ("window_size", int(args.window_time * args.imu_freq)),
            ("future_data_size", int(args.future_time * args.imu_freq)),
            ("step_size", int(args.imu_freq / args.sample_freq)),
        ]
    )
    net_config = {
        "in_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 4
    }

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Sample frequency: %s" % args.sample_freq)
    return data_window_config, net_config


def net_test(args):
    """
    Main function for network testing
    Generate trajectories, plots, and metrics.json file
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.test_list is None:
            raise ValueError("test_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            logging.info(f"Testing output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    test_list = get_datalist(args.test_list)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {args.model_path} loaded to device {device}.")

    # initialize containers
    all_metrics = {}

    for data in test_list:
        logging.info(f"Processing {data}...")
        try:
            seq_dataset = FbSequenceDataset(
                args.root_dir, [data], args, data_window_config, mode="test"
            )
            seq_loader = DataLoader(seq_dataset, batch_size=128, shuffle=False)
        except OSError as e:
            print(e)
            continue

        # Obtain trajectory
        net_attr_dict = get_inference(network, seq_loader, device, epoch=50,args=args)
        traj_attr_dict = vel_integrate(args, seq_dataset, net_attr_dict["preds"])
        outdir = osp.join(args.out_dir, data)
        if osp.exists(outdir) is False:
            os.mkdir(outdir)
        outfile = osp.join(outdir, "trajectory.txt")
        trajectory_data = np.concatenate(
            [
                traj_attr_dict["ts"].reshape(-1, 1),
                traj_attr_dict["vel_pred"],
                traj_attr_dict["vel_gt"],
            ],
            axis=1,
        )
        np.savetxt(outfile, trajectory_data, delimiter=",")

        # obtain metrics
        metrics, plot_dict = compute_metrics_and_plotting(
            args, net_attr_dict, traj_attr_dict
        )
        logging.info(metrics)
        all_metrics[data] = metrics

        outfile_net = osp.join(outdir, "net_outputs.txt")
        net_outputs_data = np.concatenate(
            [
                plot_dict["pred_ts"].reshape(-1, 1),
                plot_dict["preds"],
                plot_dict["targets"],
            ],
            axis=1,
        )
        np.savetxt(outfile_net, net_outputs_data, delimiter=",")

        if args.save_plot:
            make_plots(args, plot_dict, outdir)

        try:
            with open(args.out_dir + "/metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=1)
        except ValueError as e:
            raise e
        except OSError as e:
            print(e)
            continue
        except Exception as e:
            raise e

    return
