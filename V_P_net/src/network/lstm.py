import time

import torch
from torch import nn
import math
import torch.nn.functional as F
from pyquaternion import Quaternion
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.

class world_v_lstm(nn.Module):

    def __init__(self, input_dim, hidden_dim,num_layer,out_dim=3):
        super(world_v_lstm, self).__init__()
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim

        self.input = nn.GRU(input_dim, hidden_dim[0],num_layers=num_layer,bidirectional=False)
        self.layer_1 = nn.GRU(hidden_dim[0], hidden_dim[1], num_layers=num_layer, bidirectional=False)
        self.layer_2 = nn.GRU(hidden_dim[1] * 1, hidden_dim[2], num_layers=num_layer, bidirectional=False)
        self.layer_3 = nn.GRU(hidden_dim[2] * 1, hidden_dim[3], num_layers=num_layer, bidirectional=False)

        self.output = nn.Sequential(nn.Linear(hidden_dim[3] * 1,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(128,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(128,out_dim))
        self.output_conv = nn.Sequential(nn.Linear(hidden_dim[3] * 1,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(128,128),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5),
                                    nn.Linear(128,out_dim))

    def forward(self, x,hidden_state=None,a=None):
        x, self.hidden = self.input(x,hidden_state)
        x, self.hidden = self.layer_1(x)
        x, self.hidden = self.layer_2(x)
        x, self.hidden = self.layer_3(x)

        out = self.output(x)
        out_conv = self.output_conv(x)
        return out,out_conv

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class world_p_v_lstm_inte_axis(nn.Module):
    def __init__(self, input_dim, num_layer,hidden_dim,out_dim):
        super(world_p_v_lstm_inte_axis, self).__init__()
        self.v_model = world_v_lstm(input_dim=input_dim,hidden_dim=hidden_dim,num_layer=num_layer,out_dim=out_dim)
        self.p_model = world_v_lstm(input_dim=input_dim, hidden_dim=hidden_dim, num_layer=num_layer,out_dim=out_dim)

    def forward(self, x,t,dt,a,args,k,v0):
        x = x[:,:,[k,3]]
        v,v_cov = self.v_model(x,None)
        gravity = torch.tensor([[[0], [0], [args.gravity]]]).repeat((a.shape[0], 1, 1)).to(a.device)
        delta_p = 0.5 * (a + gravity) * dt * dt
        p_inte = (v + v0[:,:,k:k+1]) * t.unsqueeze(1).unsqueeze(1) - torch.sum(delta_p[:, :, 1:], axis=2).unsqueeze(1)[:,:,k:k+1]
        feat_p_inte_t = torch.cat((p_inte, t.unsqueeze(1).unsqueeze(1)), dim=2)
        p, p_cov = self.p_model(feat_p_inte_t, None)

        return torch.cat((p,v),dim=2),torch.cat((p_cov,v_cov),dim=2)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

