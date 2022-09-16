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

import numpy as np


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()

def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


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


def get_inference(network, data_loader, device, epoch, args,data_name):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """

    targs_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    network.eval()
    arch = args.arch

    for bid, (feat, targ_gt_v, trans_t, ts_win, gt_v_all,gt_p,gt_p_all,time,ori_r,q_inte,seq_id, frame_id) in enumerate(data_loader):

        v0 = targ_gt_v[0,:].unsqueeze(0).unsqueeze(0)
        p0 = gt_p[0,:].unsqueeze(0).unsqueeze(0)

        if arch in ["world_p_v_lstm_axis"]:

            net_dirs = [args.x_model,args.y_model,args.z_model]

            for k in range(len(net_dirs)):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                checkpoint = torch.load(net_dirs[k],map_location=device)
                network.load_state_dict(checkpoint["model_state_dict"])
                network.eval()
                if len(ts_win.shape) != 3:
                    ts_win = ts_win.unsqueeze(1)
                ori_r = q2r(q_inte)
                a_world = torch.einsum("atip,atp->ati", ori_r, feat[:, 3:6].permute(0, 2, 1)).permute(0, 2, 1)
                v_inte = jifen_v(ts_win, a_world, device, args).unsqueeze(1)
                feat_v_inte_t = torch.cat((v_inte.to(device), trans_t.unsqueeze(1).unsqueeze(1).to(device)), dim=2)

                pred, pred_cov = network(feat_v_inte_t.to(device),
                                         trans_t.to(device),
                                         ts_win.to(device), a_world.to(device), args, k, v0.to(device))
                pred = pred.squeeze(1)
                pred_cov = pred_cov.squeeze(1)
                targ_gt_p_v = torch.cat(
                    (gt_p_all[:, -1, :][:, k:k + 1] - p0.squeeze(1)[:, k:k + 1],
                     gt_v_all[:, -1, :][:, k:k + 1] - v0.squeeze(1)[:, k:k + 1]), dim=1)

                loss = get_loss(pred, pred_cov, targ_gt_p_v.to(device), epoch, arch)

                preds_all.append(pred)
                preds_cov_all.append(pred_cov)

            p = torch_to_numpy(torch.cat((preds_all[0][:, 0].unsqueeze(1), preds_all[1][:, 0].unsqueeze(1), preds_all[2][:, 0].unsqueeze(1)),
                                         dim=1) + p0.squeeze(1).to(device))
            p_cov = torch_to_numpy(torch.cat((torch.exp(2 * preds_cov_all[0][:, 0].unsqueeze(1)),
                                              torch.exp(2 * preds_cov_all[1][:, 0].unsqueeze(1)),
                                              torch.exp(2 * preds_cov_all[2][:, 0].unsqueeze(1))), dim=1))
            v = torch_to_numpy(torch.cat((preds_all[0][:, 1].unsqueeze(1), preds_all[1][:, 1].unsqueeze(1), preds_all[2][:, 1].unsqueeze(1)),
                                         dim=1) + v0.squeeze(1).to(device))
            v_cov = torch_to_numpy(torch.cat((torch.exp(2 * preds_cov_all[0][:, 1].unsqueeze(1)),
                                              torch.exp(2 * preds_cov_all[1][:, 1].unsqueeze(1)),
                                              torch.exp(2 * preds_cov_all[2][:, 1].unsqueeze(1))), dim=1))

            xlb = 'ts'
            ylbs = ['x','y','z']
            ts = np.expand_dims(np.array(time)[:, -1], axis=1)
            dpi = 90
            figsize = (16, 9)
            fig1 = plt.figure(dpi=dpi, figsize=figsize)
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts,v[:, i],label="pred_imu_v_in_world_frame")
                plt.plot(ts,torch_to_numpy(gt_v_all[:, -1, :])[:, i],label="gt_imu_v_in_world_frame")
                plt.ylabel(ylbs[i])
                plt.legend()
            plt.xlabel(xlb)
            #fig1.show()

            dpi = 90
            figsize = (16, 9)
            fig2 = plt.figure(dpi=dpi, figsize=figsize)
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts,p[:, i],label="pred_imu_p_in_world_frame")
                plt.plot(ts,torch_to_numpy(gt_p_all[:, -1, :])[:, i],label="gt_imu_p_in_world_frame")
                plt.ylabel(ylbs[i])
                plt.legend()
            plt.xlabel(xlb)
            #fig2.show()

            dpi = 90
            figsize = (16, 9)
            fig3 = plt.figure(dpi=dpi, figsize=figsize)
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts,np.abs(v[:, i] - torch_to_numpy(gt_v_all[:, -1, :])[:, i]),label="imu_v_in_world_frame_error")
                plt.ylabel(ylbs[i])
                plt.legend()
            plt.xlabel(xlb)
            #fig3.show()

            dpi = 90
            figsize = (16, 9)
            fig4 = plt.figure(dpi=dpi, figsize=figsize)
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts,np.abs(p[:, i] - torch_to_numpy(gt_p_all[:, -1, :])[:, i]),label="imu_p_in_world_frame_error")
                plt.ylabel(ylbs[i])
                plt.legend()
            plt.xlabel(xlb)
            #fig4.show()

            dpi = 90
            figsize = (16, 9)
            fig5 = plt.figure(dpi=dpi, figsize=figsize)
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(ts,v_cov[:, i],label="imu_v_cov_in_world_frame")
                plt.ylabel(ylbs[i])
                plt.legend()
            plt.xlabel(xlb)
            #fig5.show()

            dpi = 90
            figsize = (16, 9)
            fig6 = plt.figure(dpi=dpi, figsize=figsize)
            for i in range(3):
                plt.subplot(3, 1, i + 1)
                plt.plot(p_cov[:, i],label="imu_p_cov_in_world_frame")
                plt.ylabel(ylbs[i])
                plt.legend()
            plt.xlabel(xlb)
            #fig6.show()

            data_vp = np.concatenate((ts, v, p), axis=1)
            data_vp_cov = np.concatenate((ts, v_cov, p_cov), axis=1)
            np.savetxt(args.out_dir + '/vp/' +  data_name + '_vp.txt', data_vp, delimiter=',')
            np.savetxt(args.out_dir + '/vp_cov/' +  data_name + '_vp_cov.txt', data_vp_cov,delimiter=',')

            fig1.savefig(args.out_dir + '/imu_v_in_world_frame/' + data_name + '_imu_v_in_world_frame.png')
            fig2.savefig(args.out_dir + '/imu_p_in_world_frame/' + data_name + '_imu_p_in_world_frame.png')
            fig3.savefig(args.out_dir + '/imu_v_error_in_world_frame/' + data_name + '_imu_v_error_in_world_frame.png')
            fig4.savefig(args.out_dir + '/imu_p_error_in_world_frame/' + data_name + '_imu_p_error_in_world_frame.png')
            fig5.savefig(args.out_dir + '/imu_v_cov_in_world_frame/' + data_name + '_imu_v_cov_in_world_frame.png')
            fig6.savefig(args.out_dir + '/imu_p_cov_in_world_frame/' + data_name + '_imu_p_cov_in_world_frame.png')

            print('a')
        else:
            raise ValueError("Invalid architecture to train.py:", arch)


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
            if not osp.isdir(args.out_dir + '/'+'imu_p_cov_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_p_cov_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_p_cov_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_p_error_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_p_error_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_p_error_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_p_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_p_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_p_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_v_cov_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_v_cov_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_v_cov_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_v_error_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_v_error_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_v_error_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'imu_v_in_world_frame'):
                os.makedirs(args.out_dir + '/'+'imu_v_in_world_frame')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'imu_v_in_world_frame'}")
            if not osp.isdir(args.out_dir + '/'+'vp'):
                os.makedirs(args.out_dir + '/'+'vp')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'vp'}")
            if not osp.isdir(args.out_dir + '/'+'vp_cov'):
                os.makedirs(args.out_dir + '/'+'vp_cov')
                logging.info(f"Testing output writes to {args.out_dir+ '/'+'vp_cov'}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )

    # initialize containers
    all_metrics = {}

    test_list = get_datalist(args.test_list)
    train_loader_list = []
    for data in test_list:
        data = [data]
        try:
            seq_dataset = FbSequenceDataset(
                args.root_dir, data, args, data_window_config, mode="test"
            )
            seq_loader = DataLoader(
                seq_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False  # 根据batch_size加载数据，并根据shuffle对数据打乱 # 这里把两个都改成false了
            )
        except OSError as e:
            logging.error(e)
            return
        # Obtain trajectory
        get_inference(network, seq_loader, device, 500000000 ,args, data[0])


    return
