"""
This file includes the main libraries in the network training module.
"""

import json
import os
import signal
import sys
import time
from functools import partial
from os import path as osp

import numpy as np
import torch
from dataloader.dataset_fb import FbSequenceDataset
from network.model_factory import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.logging import logging

from torch.nn import functional as F
from network.losses import get_loss, qmul

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


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

def jifen_q(gyr, ts_win, args):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """
    gyr = gyr.permute(0, 2, 1)
    ts_win = ts_win.permute(0, 2, 1)
    feat_gyr = gyr

    # 补齐10个 d_t 数据
    dalte_w = feat_gyr
    dalte_w_norm = torch.norm(dalte_w, p=2, dim=2, keepdim=True)
    dalte_intint = dalte_w_norm * ts_win / 2

    w_point = dalte_w / dalte_w_norm
    dalte_q_w = torch.cos(dalte_intint)
    dalte_q_xyz = w_point * torch.sin(dalte_intint)
    dalte_q_wxyz = torch.cat((dalte_q_w, dalte_q_xyz), dim=2)

    q_inte = dalte_q_wxyz[:, 0, :]
    for i in range(1, dalte_q_wxyz.shape[1]):
        q_inte = qmul(q_inte, dalte_q_wxyz[:, i, :])
    q_inte = F.normalize(q_inte)
    q_inte = torch.where((q_inte[:, 0] > 0).reshape(-1, 1), q_inte, -q_inte)

    return q_inte


def get_inference(network, data_loader, device, epoch, args):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    arch = args.arch
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    network.eval()

    for bid, (feat_target, _, _) in enumerate(data_loader):
        feat = feat_target["feat"]
        gamma_q_win = feat_target["gamma_q_win"]
        d_t = feat_target["d_t"]
        # 判断是否包含past_time, 然后计算窗口的 q_win_integral
        if args.past_time + args.window_time + args.future_time != args.window_time:
            p_size = int(args.past_time * args.imu_freq)
            w_size = int(args.window_time * args.imu_freq)
            gyr_feat = feat[:, :3, p_size:p_size+w_size]  # 带有bias的原始数据
            d_t_win = d_t[:, p_size:p_size+w_size].unsqueeze(1)
        elif args.past_time + args.window_time + args.future_time == args.window_time:
            gyr_feat = feat[:, :3, :]  # 带有bias的原始数据
            d_t_win = d_t.unsqueeze(1)
        else:
            raise ValueError("!!!! wrong calculate gyr")

        if arch in [ "resnet"]:

            feat = feat[:, :3, :] #
            pred = network(feat.to(device))

            gyr_feat_offset = (gyr_feat.to(device) + pred.unsqueeze(2).repeat(1, 1, gyr_feat.shape[2]))
            q_win_offset_integral = jifen_q(gyr_feat_offset, d_t_win.to(device), args)
            q_win_offset_integral_norm = F.normalize(q_win_offset_integral, dim=len(q_win_offset_integral.shape) - 1)
            q_win_offset_integral_norm[q_win_offset_integral_norm[:, 0] < 0] = (-1) * q_win_offset_integral_norm[q_win_offset_integral_norm[:, 0] < 0]
            
            gamma_q_win_norm = F.normalize(gamma_q_win, dim=len(gamma_q_win.shape) - 1)
            gamma_q_win_norm[gamma_q_win_norm[:, 0] < 0] = (-1) * gamma_q_win_norm[gamma_q_win_norm[:, 0] < 0]
            gamma_q_win_norm = gamma_q_win_norm.to(q_win_offset_integral_norm.device)

            loss = get_loss(q_win_offset_integral_norm,gamma_q_win_norm, arch)

            targets_all.append(torch_to_numpy(gamma_q_win_norm))
            preds_all.append(torch_to_numpy(q_win_offset_integral_norm))
            losses_all.append(torch_to_numpy(loss))

        else:
            raise ValueError("Invalid architecture to train.py:", arch)

    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)

    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "losses": losses_all,
    }
    return attr_dict


def do_train(network, train_loader, device, epoch, optimizer, args):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    arch = args.arch
    train_targets, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()

    for bid, (feat_target, _, _) in enumerate(train_loader):
        feat = feat_target["feat"]
        gamma_q_win = feat_target["gamma_q_win"]
        d_t = feat_target["d_t"]

        if args.past_time + args.window_time + args.future_time != args.window_time:
            p_size = int(args.past_time * args.imu_freq)
            w_size = int(args.window_time * args.imu_freq)
            gyr_feat = feat[:, :3, p_size:p_size+w_size]
            d_t_win = d_t[:, p_size:p_size+w_size].unsqueeze(1)
        elif args.past_time + args.window_time + args.future_time == args.window_time:
            gyr_feat = feat[:, :3, :]
            d_t_win = d_t.unsqueeze(1)
        else:
            raise ValueError("!!!! wrong calculate gyr")

        optimizer.zero_grad()

        if arch in [ "resnet"]:
            feat = feat[:, :3, :]
            pred = network(feat.to(device))

            gyr_feat_offset = (gyr_feat.to(device) + pred.unsqueeze(2).repeat(1, 1, gyr_feat.shape[2]))
            q_win_offset_integral = jifen_q(gyr_feat_offset, d_t_win.to(device), args)
            q_win_offset_integral_norm = F.normalize(q_win_offset_integral, dim=len(q_win_offset_integral.shape) - 1)
            q_win_offset_integral_norm[q_win_offset_integral_norm[:, 0] < 0] = (-1) * q_win_offset_integral_norm[q_win_offset_integral_norm[:, 0] < 0]
            
            gamma_q_win_norm = F.normalize(gamma_q_win, dim=len(gamma_q_win.shape) - 1)
            gamma_q_win_norm[gamma_q_win_norm[:, 0] < 0] = (-1) * gamma_q_win_norm[gamma_q_win_norm[:, 0] < 0]
            gamma_q_win_norm = gamma_q_win_norm.to(q_win_offset_integral_norm.device)

            loss = get_loss(q_win_offset_integral_norm,gamma_q_win_norm, arch)

            train_targets.append(torch_to_numpy(gamma_q_win_norm))
            train_preds.append(torch_to_numpy(q_win_offset_integral_norm))
            train_losses.append(torch_to_numpy(loss))
        else:
            raise ValueError("Invalid architecture to train.py:", arch)

        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

    train_targets = np.concatenate(train_targets, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)

    train_attr_dict = {
        "targets": train_targets,
        "preds": train_preds,
        "losses": train_losses,
    }
    return train_attr_dict

def write_summary(summary_writer, attr_dict, epoch, optimizer, mode, arch):
    """ Given the attr_dict write summary and log the losses """
    losses = np.mean(attr_dict["losses"], axis=0)
    if losses.shape[0] == 4:
        summary_writer.add_scalar(f"{mode}_loss/loss_mse_SO3_w", losses[0], epoch)
        summary_writer.add_scalar(f"{mode}_loss/loss_mse_SO3_x", losses[1], epoch)
        summary_writer.add_scalar(f"{mode}_loss/loss_mse_SO3_y", losses[2], epoch)
        summary_writer.add_scalar(f"{mode}_loss/loss_mse_SO3_z", losses[3], epoch)
    elif losses.shape[0] == 3:
        summary_writer.add_scalar(f"{mode}_loss/loss_so3_x", losses[0], epoch)
        summary_writer.add_scalar(f"{mode}_loss/loss_so3_y", losses[1], epoch)
        summary_writer.add_scalar(f"{mode}_loss/loss_so3_z", losses[2], epoch)

    ml_loss = np.average(attr_dict["losses"])
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)
    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
        )
    logging.info(
        f"{mode}: average ml loss: {ml_loss}, average mse loss: {losses}/{np.mean(losses)}"
    )

def save_model(args, epoch, network, optimizer, interrupt=False):
    if interrupt:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
    else:
        model_path = osp.join(args.out_dir, "checkpoints", "checkpoint_%d.pt" % epoch)
    state_dict = {
        "model_state_dict": network.state_dict(),
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(state_dict, model_path)
    logging.info(f"Model saved to {model_path}")


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

    return data_window_config, net_config

def net_train(args):
    """
    Main function for network training
    """
    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.train_list is None:
            raise ValueError("train_list must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            if not osp.isdir(osp.join(args.out_dir, "checkpoints")):
                os.makedirs(osp.join(args.out_dir, "checkpoints"))
            if not osp.isdir(osp.join(args.out_dir, "logs")):
                os.makedirs(osp.join(args.out_dir, "logs"))
            with open(
                os.path.join(args.out_dir, "parameters.json"), "w"
            ) as parameters_file:
                parameters_file.write(json.dumps(vars(args), sort_keys=True, indent=4))
            logging.info(f"Training output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        if args.val_list is None:
            logging.warning("val_list is not specified.")
        if args.continue_from is not None:
            if osp.exists(args.continue_from):
                logging.info(
                    f"Continue training from existing model {args.continue_from}"
                )
            else:
                raise ValueError(
                    f"continue_from model file path {args.continue_from} does not exist"
                )
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

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

    train_loader, val_loader = None, None
    start_t = time.time()
    train_list = get_datalist(args.train_list)
    try:
        train_dataset = FbSequenceDataset(
            args.root_dir, train_list, args, data_window_config, mode="train"
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
    except OSError as e:
        logging.error(e)
        return
    end_t = time.time()
    logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
    logging.info(f"Number of train samples: {len(train_dataset)}")

    if args.val_list is not None:
        val_list = get_datalist(args.val_list)
        try:
            val_dataset = FbSequenceDataset(
                args.root_dir, val_list, args, data_window_config, mode="val"
            )
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)
        except OSError as e:
            logging.error(e)
            return
        logging.info("Validation set loaded.")
        logging.info(f"Number of val samples: {len(val_dataset)}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )
    total_params = network.get_num_params()
    logging.info(f'Network "{args.arch}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12
    )
    logging.info(f"Optimizer: {optimizer}, Scheduler: {scheduler}")

    start_epoch = 0
    if args.continue_from is not None:
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get("epoch", 0)
        network.load_state_dict(checkpoints.get("model_state_dict"))
        optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
        logging.info(f"Continue from epoch {start_epoch}")
    else:
        # default starting from latest checkpoint from interruption
        latest_pt = os.path.join(args.out_dir, "checkpoints", "checkpoint_latest.pt")
        if os.path.isfile(latest_pt):
            checkpoints = torch.load(latest_pt)
            start_epoch = checkpoints.get("epoch", 0)
            network.load_state_dict(checkpoints.get("model_state_dict"))
            optimizer.load_state_dict(checkpoints.get("optimizer_state_dict"))
            logging.info(
                f"Detected saved checkpoint, starting from epoch {start_epoch}"
            )

    summary_writer = SummaryWriter(osp.join(args.out_dir, "logs"))
    summary_writer.add_text("info", f"total_param: {total_params}")

    logging.info(f"-------------- Init, Epoch {start_epoch} --------------")
    # todo 前向输出
    attr_dict = get_inference(network, train_loader, device, start_epoch, args)
    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train", args.arch)
    if val_loader is not None:
        attr_dict = get_inference(network, val_loader, device, start_epoch, args)
        write_summary(summary_writer, attr_dict, start_epoch, optimizer, "val", args.arch)

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, interrupt=True)
        sys.exit()

    best_val_loss = np.inf
    for epoch in range(start_epoch + 1, args.epochs):
        signal.signal(
            signal.SIGINT, partial(stop_signal_handler, args, epoch, network, optimizer)
        )
        signal.signal(
            signal.SIGTERM,
            partial(stop_signal_handler, args, epoch, network, optimizer),
        )

        logging.info(f"-------------- Training, Epoch {epoch} ---------------")
        start_t = time.time()
        # todo 训练输出
        train_attr_dict = do_train(network, train_loader, device, epoch, optimizer, args)
        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train", args.arch)
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if val_loader is not None:
            val_attr_dict = get_inference(network, val_loader, device, epoch, args)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val", args.arch)
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer)
        else:
            save_model(args, epoch, network, optimizer)

    logging.info("Training complete.")

    return
