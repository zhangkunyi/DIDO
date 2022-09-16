"""
This file includes the main libraries in the network training module.
"""

import json
import os
import random
import signal
import sys
import time
from functools import partial
from os import path as osp
from liegroups.torch.so3 import SO3Matrix
from numpy.core.fromnumeric import size
from pyquaternion import Quaternion
import matplotlib.pyplot as plt

import numpy as np
import torch
from dataloader.dataset_fb import FbSequenceDataset
from network.losses import get_loss,loss_mse_so3_q
from network.model_factory import get_model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.logging import logging
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils import clip_grad_norm_
from .losses import q_log_torch
from torch import nn
from torch.nn import functional as F
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

summary_writer = SummaryWriter("../train_output/logs")

def jifen_v(dt,acc_inter,device,args):
    if len(acc_inter.shape) == 4:  # acc_inter [ seq，batch， ， ] 所以如果这个seq是被pad的，那么shape[2]对应的都是0
        dt_temp = dt.to(device)
        acc_inter_temp = acc_inter.to(device)
        acc_inter_temp = acc_inter_temp[:, :,:, int(args.past_time * args.imu_freq):].to(device)
        gravity = torch.tensor([[[[0, 0, args.gravity]]]]).repeat((acc_inter.shape[0],acc_inter.shape[1], acc_inter.shape[2], 1)).to(device)  # blackbird是 +9.8
        # dt = np.expand_dims(dt, 1).repeat(3, axis=1)
        acc_no_g = acc_inter_temp[:, :, :,:].to(device) + gravity
        temp = (acc_no_g.shape[0],acc_no_g.shape[1],acc_no_g.shape[2],1)
        acc_no_g = torch.where((acc_no_g[:,:,:,0] == 0).reshape(temp) * (acc_no_g[:,:,:,1] == 0).reshape(temp) * (acc_no_g[:,:,:,2] == -9.8).reshape(temp), torch.full_like(acc_no_g,0), acc_no_g)
        delta_v = acc_no_g * dt_temp.repeat(1,acc_inter.shape[1],1).unsqueeze(3)
        delta_v_integrate = torch.sum(delta_v[:, :, :,:], axis=2)
    else:
        dt_temp = dt.to(device)
        if acc_inter.shape[2] == 3:
            acc_inter_temp = acc_inter.permute(0,2,1).to(device)
        else:
            acc_inter_temp = acc_inter.to(device)
        acc_inter_temp = acc_inter_temp[:,:,int(args.past_time*args.imu_freq):].to(device)
        gravity = torch.tensor([[[0], [0], [args.gravity]]]).repeat((acc_inter.shape[0],1,1)).to(device) # blackbird是 +9.8
        # dt = np.expand_dims(dt, 1).repeat(3, axis=1)
        delta_v = ( acc_inter_temp[:, :, :].to(device) + gravity ) * dt_temp[:, :, :]
        delta_v_integrate = torch.sum(delta_v[:,:,:], axis=2)
    return delta_v_integrate

def q2r(q):
    if len(q.shape) == 3:
        r = torch.zeros(q.shape[0],q.shape[1],3,3).to(q.device)
        r[:,:,0,0] = 1-2*q[:,:,2]**2-2*q[:,:,3]**2
        r[:,:,0,1] = 2*q[:,:,1]*q[:,:,2] - 2*q[:,:,0]*q[:,:,3]
        r[:,:,0,2] = 2*q[:,:,1]*q[:,:,3] + 2*q[:,:,0]*q[:,:,2]
        r[:,:,1,0] = 2*q[:,:,1]*q[:,:,2] + 2*q[:,:,0]*q[:,:,3]
        r[:,:, 1, 1] = 1-2*q[:,:,1]**2-2*q[:,:,3]**2
        r[:,:, 1, 2] = 2 * q[:, :,2] * q[:, :,3] - 2 * q[:, :,0] * q[:, :,1]
        r[:,:, 2, 0] = 2 * q[:, :,1] * q[:, :,3] - 2 * q[:, :,0] * q[:, :,2]
        r[:,:, 2, 1] = 2 * q[:, :,2] * q[:, :,3] + 2 * q[:, :,0] * q[:, :,1]
        r[:,:, 2, 2] = 1 - 2 * q[:, :,1]**2 - 2 * q[:, :,2]**2

    elif len(q.shape) == 4:
        r = torch.zeros(q.shape[0],q.shape[1],q.shape[2],3,3).to(q.device)
        r[:,:,:,0,0] = 1-2*q[:,:,:,2]**2-2*q[:,:,:,3]**2
        r[:,:,:,0,1] = 2*q[:,:,:,1]*q[:,:,:,2] - 2*q[:,:,:,0]*q[:,:,:,3]
        r[:,:,:,0,2] = 2*q[:,:,:,1]*q[:,:,:,3] + 2*q[:,:,:,0]*q[:,:,:,2]
        r[:,:,:,1,0] = 2*q[:,:,:,1]*q[:,:,:,2] + 2*q[:,:,:,0]*q[:,:,:,3]
        r[:,:, :,1, 1] = 1-2*q[:,:,:,1]**2-2*q[:,:,:,3]**2
        r[:,:,:, 1, 2] = 2 * q[:, :,:,2] * q[:, :,:,3] - 2 * q[:, :,:,0] * q[:, :,:,1]
        r[:,:,:, 2, 0] = 2 * q[:, :,:,1] * q[:, :,:,3] - 2 * q[:, :,:,0] * q[:, :,:,2]
        r[:,:, :,2, 1] = 2 * q[:, :,:,2] * q[:, :,:,3] + 2 * q[:, :,:,0] * q[:, :,:,1]
        r[:,:, :,2, 2] = 1 - 2 * q[:, :,:,1]**2 - 2 * q[:, :,:,2]**2

    return r

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list

def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(network, data_loader_list, device, epoch, args):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """

    targs_all, preds_all, preds_cov_all, losses_all,vs_all = [], [], [], [],[]
    network.eval()
    arch = args.arch

    for data_loader in data_loader_list:
        for bid, (feat, targ_gt_v, trans_t, ts_win, gt_v_all,gt_p,gt_p_all,time,ori_r,q_inte,seq_id, frame_id) in enumerate(data_loader):

            v0 = targ_gt_v[0,:].unsqueeze(0).unsqueeze(0)
            p0 = gt_p[0,:].unsqueeze(0).unsqueeze(0)

            if arch in ["world_p_v_lstm_axis"]:

                if args.train_axis == 'x_axis':
                    k = 0
                elif args.train_axis == 'y_axis':
                    k = 1
                elif args.train_axis == 'z_axis':
                    k = 2
                else:
                    raise ValueError("Invalid Train_axis: ", arch)

                ts_win = ts_win.unsqueeze(1)
                ori_r  = q2r(q_inte)
                a_world = torch.einsum("atip,atp->ati", ori_r , feat[:, 3:6].permute(0, 2, 1)).permute(0,2,1)
                v_inte = jifen_v(ts_win, a_world, device, args).unsqueeze(1)
                feat_v_inte_t = torch.cat((v_inte.to(device), trans_t.unsqueeze(1).unsqueeze(1).to(device)), dim=2)

                pred, pred_cov = network(feat_v_inte_t.to(device),
                                         trans_t.to(device),
                                         ts_win.to(device), a_world.to(device), args ,k,v0.to(device))
                pred = pred.squeeze(1)
                pred_cov = pred_cov.squeeze(1)
                targ_gt_p_v = torch.cat(
                    (gt_p_all[:, -1, :][:,k:k+1] - p0.squeeze(1)[:,k:k+1], gt_v_all[:, -1, :][:,k:k+1] - v0.squeeze(1)[:,k:k+1]), dim=1)
                loss = get_loss(pred, pred_cov, targ_gt_p_v.to(device), epoch, arch)
                targs_all.append(torch_to_numpy(targ_gt_p_v))
            else:
                raise ValueError("Invalid architecture to train.py:", arch)

            preds_all.append(torch_to_numpy(pred))
            preds_cov_all.append(torch_to_numpy(pred_cov))
            losses_all.append(torch_to_numpy(loss))


    targs_all = np.concatenate(targs_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    preds_cov_all = np.concatenate(preds_cov_all, axis=0)
    losses_all = np.concatenate(losses_all, axis=0)

    attr_dict = {
        "targs": targs_all,
        "preds": preds_all,
        "preds_cov": preds_cov_all,
        "losses": losses_all,
    }

    return attr_dict

def do_train(network, data_loader_list, device, epoch, optimizer,args):
    """
    Train network for one epoch using a specified data loader
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    """
    train_targs, train_preds, train_preds_cov, train_losses = [], [], [], []
    network.train()
    arch = args.arch

    for data_loader in data_loader_list:
        for bid, (feat, targ_gt_v, trans_t, ts_win, gt_v_all,gt_p,gt_p_all,time,ori_r,q_inte,seq_id, frame_id) in enumerate(data_loader):

                v0 = targ_gt_v[0, :].unsqueeze(0).unsqueeze(0)
                p0 = gt_p[0, :].unsqueeze(0).unsqueeze(0)

                optimizer.zero_grad()

                if arch in ["world_p_v_lstm_axis"]:

                    if args.train_axis == 'x_axis':
                        k = 0
                    elif args.train_axis == 'y_axis':
                        k = 1
                    elif args.train_axis == 'z_axis':
                        k = 2

                    ts_win = ts_win.unsqueeze(1)
                    a_world = torch.einsum("atip,atp->ati", ori_r, feat[:, 3:6].permute(0, 2, 1)).permute(0,2,1)

                    v_inte = jifen_v(ts_win, a_world, device, args).unsqueeze(1)
                    feat_v_inte_t = torch.cat((v_inte.to(device), trans_t.unsqueeze(1).unsqueeze(1).to(device)), dim=2)

                    pred, pred_cov = network(feat_v_inte_t.to(device),
                                             trans_t.to(device), ts_win.to(device), a_world.to(device),
                                             args ,k,v0.to(device))
                    pred = pred.squeeze(1)
                    pred_cov = pred_cov.squeeze(1)
                    targ_gt_p_v = torch.cat(
                        (gt_p_all[:, -1, :][:,k:k+1] - p0.squeeze(1)[:,k:k+1], gt_v_all[:, -1, :][:,k:k+1] - v0.squeeze(1)[:,k:k+1]), dim=1)
                    loss = get_loss(pred, pred_cov, targ_gt_p_v.to(device), epoch, arch)
                    train_targs.append(torch_to_numpy(targ_gt_p_v))
                else:
                    raise ValueError("Invalid architecture to train.py:", arch)

                train_preds.append(torch_to_numpy(pred))
                train_preds_cov.append(torch_to_numpy(pred_cov))
                train_losses.append(torch_to_numpy(loss))

                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()


    train_targs = np.concatenate(train_targs, axis=0)
    train_preds = np.concatenate(train_preds, axis=0)
    train_preds_cov = np.concatenate(train_preds_cov, axis=0)
    train_losses = np.concatenate(train_losses, axis=0)
    train_attr_dict = {
        "targs": train_targs,
        "preds": train_preds,
        "preds_cov": train_preds_cov,
        "losses": train_losses,
    }

    return train_attr_dict

def loss_mse_R3(pred, targ):
    loss = (pred - targ).pow(2)
    return loss

def write_summary(summary_writer, attr_dict, epoch, optimizer, mode,args):
    """ Given the attr_dict write summary and log the losses """
    arch = args.arch
    if arch in ["world_p_v_lstm_axis"]:
        mse_loss_R = np.mean((attr_dict["targs"][:, :] - attr_dict["preds"][:, :]) ** 2, axis=0)
        summary_writer.add_scalar(f"{mode}_loss/loss_p", mse_loss_R[0], epoch)
        summary_writer.add_scalar(f"{mode}_loss/loss_v", mse_loss_R[1], epoch)
    else:
        raise ValueError("Invalid architecture to train.py:", arch)

    ml_loss = np.average(attr_dict["losses"])
    summary_writer.add_scalar(f"{mode}_dist/loss_full", ml_loss, epoch)

    sigmas = np.exp(attr_dict["preds_cov"])
    if arch in  ["world_p_v_lstm_axis"]:
        summary_writer.add_histogram(f"{mode}_hist/sigma_px", sigmas[:, 0], epoch)
        summary_writer.add_histogram(f"{mode}_hist/sigma_vy", sigmas[:, 1], epoch)

    if epoch > 0:
        summary_writer.add_scalar(
            "optimizer/lr", optimizer.param_groups[0]["lr"], epoch - 1
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
    torch.save(state_dict, model_path,_use_new_zipfile_serialization=True)

    logging.info(f"Model saved to {model_path}")


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer(): # 判断 过去时间*imu_freq 是否是整数 (imu数据输入size必须为整数)
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
        data_window_config, net_config = arg_conversion(args)  # 得到设置的数据窗口大小，imu输入的网络参数
    except ValueError as e:
        logging.error(e)
        return

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])   # 打印输入数据的size和选取的时间长度
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
    train_loader_list = []
    for data in train_list:
        data = [data]
        try:
            train_dataset = FbSequenceDataset(
                args.root_dir, data, args, data_window_config, mode="train"
            )
            # train_loader = DataLoader(
            #     train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=my_collate  # 根据batch_size加载数据，并根据shuffle对数据打乱 # 这里把两个都改成false了
            # )
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False  # 根据batch_size加载数据，并根据shuffle对数据打乱 # 这里把两个都改成false了
            )
        except OSError as e:
            logging.error(e)
            return
        end_t = time.time()
        logging.info(f"Training set loaded. Loading time: {end_t - start_t:.3f}s")
        logging.info(f"Number of train samples: {len(train_dataset)}")
        train_loader_list.append(train_loader)

    if args.val_list is not None:
        val_list = get_datalist(args.val_list)
        val_loader_list = []
        for data in val_list:
            data = [data]
            try:
                val_dataset = FbSequenceDataset(
                    args.root_dir, data, args, data_window_config, mode="val"
                )
                # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False,collate_fn=my_collate) # 这里把两个都改成false了
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            except OSError as e:
                logging.error(e)
                return
            logging.info("Validation set loaded.")
            logging.info(f"Number of val samples: {len(val_dataset)}")
            val_loader_list.append(val_loader)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )

    print("=====================network====================")
    print(network)
    print("=====================network====================")

    total_params = network.get_num_params()                            # 网络的整个参数？？？？？？？？？
    logging.info(f'Network "{args.arch}" loaded to device {device}')
    logging.info(f"Total number of parameters: {total_params}")

    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad,network.parameters()), args.lr)
    # optimizer = torch.optim.Adam( network.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(           # 学习速率调整
        optimizer, factor=0.1, patience=10, verbose=True, eps=1e-12   # eps：最小衰减适用于lr。如果新lr和旧lr之间的差小于eps，则忽略更新。
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
    attr_dict = get_inference(network, train_loader_list, device, start_epoch,args)
    write_summary(summary_writer, attr_dict, start_epoch, optimizer, "train",args)
    if val_loader_list is not None:
        attr_dict = get_inference(network, val_loader_list, device, start_epoch,args)
        write_summary(summary_writer, attr_dict, start_epoch, optimizer, "val",args)

    def stop_signal_handler(args, epoch, network, optimizer, signal, frame):
        logging.info("-" * 30)
        logging.info("Early terminate")
        save_model(args, epoch, network, optimizer, interrupt=True)
        sys.exit()

    best_val_loss = np.inf

    best_val_loss_2 = np.inf
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
        train_attr_dict = do_train(network, train_loader_list, device, epoch, optimizer,args)
        write_summary(summary_writer, train_attr_dict, epoch, optimizer, "train",args)
        end_t = time.time()
        logging.info(f"time usage: {end_t - start_t:.3f}s")

        if val_loader_list is not None:
            val_attr_dict = get_inference(network, val_loader_list, device, epoch,args)
            write_summary(summary_writer, val_attr_dict, epoch, optimizer, "val",args)

            if np.mean((val_attr_dict["targs"] - val_attr_dict["preds"]) ** 2) < best_val_loss_2:
                best_val_loss_2 = np.mean((val_attr_dict["targs"] - val_attr_dict["preds"]) ** 2)
                save_model(args, epoch, network, optimizer)
            if np.mean(val_attr_dict["losses"]) < best_val_loss:
                best_val_loss = np.mean(val_attr_dict["losses"])
                save_model(args, epoch, network, optimizer)

        else:
            save_model(args, epoch, network, optimizer)

    logging.info("Training complete.")

    return
