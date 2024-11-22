import torch
import math
import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from mpl_toolkits.mplot3d import Axes3D

class Data_set(torch.utils.data.Dataset):
    def __init__(self, apop_intr, apop_extr):
        self.transform = T.ToTensor()
        if apop_intr is not None:
            self.intr_flag = True
            self.intr_w_coords = apop_intr.world_coords
            self.intr_p_coords = apop_intr.pixel_coords
            self.intr_i_params = apop_intr.param_idx
            self.intr_names = apop_intr.name_list
            self.intr_images = apop_intr.rgb_imgs
            assert self.intr_w_coords.shape[0] == self.intr_p_coords.shape[0] == self.intr_i_params.shape[0] == len(self.intr_names), "Intrinsic Data format Error !!"
        else:
            self.intr_flag = None
            
        if apop_extr is not None:
            self.extr_flag = True
            self.extr_w_coords = apop_extr.world_coords
            self.extr_p_coords = apop_extr.pixel_coords
            self.extr_i_params = apop_extr.param_idx
            self.extr_names = apop_extr.name_list
            self.extr_images = apop_extr.rgb_imgs
            assert self.extr_w_coords.shape[0] == self.extr_p_coords.shape[0] == self.extr_i_params.shape[0] == len(self.extr_names), "Extrinsic Data format Error !!"
        else:
            self.extr_flag = None

    def __len__(self):
        if self.intr_flag is None:
            return len(self.extr_names)
        elif self.extr_flag is None:
            return len(self.intr_names)
        else:
            assert len(self.intr_names) == len(self.extr_names), "Mismatch Between Intrinsics and Extrinsics !!"
            return len(self.intr_names)

    def __getitem__(self, idx):
        if self.intr_flag is None:
            return self.extr_w_coords[idx], self.extr_p_coords[idx], self.extr_i_params[idx], self.extr_names[idx]
        elif self.extr_flag is None:
            return self.intr_w_coords[idx], self.intr_p_coords[idx], self.intr_i_params[idx], self.intr_names[idx]
        else:
            return self.intr_w_coords[idx], self.intr_p_coords[idx], self.intr_i_params[idx], self.intr_names[idx],\
                   self.extr_w_coords[idx], self.extr_p_coords[idx], self.extr_i_params[idx], self.extr_names[idx]


    def init_show_figure(self, show_info=True):
        self.all_fig = plt.figure(figsize=(4,4))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.default'] = 'regular'
        self.ax = Axes3D(self.all_fig, auto_add_to_figure=False)
        self.all_fig.add_axes(self.ax)
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            
        if show_info:
            self.ax.set_xlabel("X Axis")
            self.ax.set_ylabel("Y Axis")
            self.ax.set_zlabel("Z Axis")
            # self.ax.set_xlim(-3.5, 3.5)
            # self.ax.set_ylim(-3.5, 3.5)
            # self.ax.set_zlim(-1.5, 3.5)
        else:
            self.ax.grid(False)
            self.ax.axis(False)

        plt.ion()
        plt.gca().set_box_aspect((1, 1, 1))



    
