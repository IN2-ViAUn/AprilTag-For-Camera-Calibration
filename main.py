import torch
import json
import cv2
import os 

import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from components import AprilTag_Operation
from dataloader import Data_set
from mpl_toolkits.mplot3d import Axes3D

class Optimization_For_Reprojection(nn.Module):
    def __init__(self, config, apop_intr, apop_extr):
        super(Optimization_For_Reprojection, self).__init__()
        self.config = config
        self.device = config["device"]
        self.opt_intr = config["intrinsics"]
        self.reproj_2d_pts_save_pth = config["reproj_save_path"]
        self.save_param_path = config["param_save_path"]
        self.apop_intr, self.apop_extr = apop_intr, apop_extr
        self.intr_numb, self.extr_numb = self.generate_params_to_learn(self.apop_intr, self.apop_extr)
        self.init_show_figure() 
        
        # load provided intrinsics
        if not self.opt_intr:
            intr_dict = self.load_intrinsics_from_json(config["intrinsics_path"])
            self.intr_fix = self.match_names_to_intrinsics(intr_dict, apop_extr)

    def forward(self, wpts):
        reproj_pts_intr = 0.0
        reproj_pts_extr = 0.0
        # create learnable parameters    
        self.intr_adj, self.pose_adj = self.add_weights2params(self.apop_intr, self.apop_extr)
        
        if not self.opt_intr:
            self.intr_adj = self.intr_fix
            
        if self.config["intrinsics"]:
            reproj_pts_intr = self.get_reproject_pixels(wpts, self.intr_adj, self.pose_adj)
        if self.config["extrinsics"]:
            reproj_pts_extr = self.get_reproject_pixels(wpts, self.intr_adj, self.pose_adj)
        
        loss_dict = {"intr": reproj_pts_intr,
                     "extr": reproj_pts_extr}
        
        return loss_dict
    

    def load_intrinsics_from_json(self, json_path):
        intr_dict = {}
        with open(json_path, 'r') as f:
            camera_params = json.load(f)        
        for cam_name, cam_data in camera_params.items():
            intr_dict[cam_name + '.png'] = np.array(cam_data["intrinsic"])
        return intr_dict

    # tag_wpts:[Batch, HW, 3]
    # intr_adj:[Batch, 3, 3]
    # pose_adj:[Batch, 3, 4]
    def get_reproject_pixels(self, tag_wpts, intr_adj, pose_adj):
        world_pts = self.world2hom(tag_wpts)
        # proj_cam_pts = self.world2cam(world_pts, pose_adj.unsqueeze(0))
        # proj_pts = self.cam2pix(proj_cam_pts, intr_adj.unsqueeze(0))
        proj_cam_pts = self.world2cam(world_pts, pose_adj)
        proj_pts = self.cam2pix(proj_cam_pts, intr_adj)

        return proj_pts

    # [Batch, HW, 3]->[Batch, HW, 4]
    def world2hom(self, world_cord):
        X_hom = torch.cat([world_cord, torch.ones_like(world_cord[...,:1])], dim=-1)
        return X_hom     

    # world_cord: [batch, ..., 4]
    # pose: [batch, ..., 3, 4]
    def world2cam(self, world_cord, pose):
        shape = pose.shape
        supply_pose = torch.tensor([0, 0, 0, 1], device=self.device)
        supply_pose = supply_pose.expand(shape)[...,:1,:]
        hom_pose = torch.cat([pose, supply_pose], dim=-2)
        cam_cord = hom_pose @ world_cord.transpose(-2, -1)

        return cam_cord

    # cam_cord: [batch, ..., 4]
    # intr_mat: [batch, ..., 3, 3]    
    def cam2pix(self, cam_cord, intr_mat):
        hom_intr_mat = torch.cat([intr_mat, torch.zeros_like(intr_mat[...,:1])], dim=-1)
        pix_cord = hom_intr_mat @ cam_cord
        pix_cord = pix_cord[...,:2,:]/pix_cord[...,2:,:]
        pix_cord = pix_cord.transpose(-2, -1)
        return pix_cord
    
    # Assign a learnable extrinsic parameter to each camera
    def match_names_to_intrinsics(self, intr_dict, apop_extr):
        new_intr_list = []
        name_list = apop_extr.name_list
        for name in name_list:
            try:
                intr = intr_dict[name]
            except ValueError:
                print("{} is not avalible, skipping...".format(name))
                continue
            new_intr_list += [torch.tensor(intr, dtype=torch.float32, device=self.device)]
        new_intr_list = torch.stack(new_intr_list, 0)

        return new_intr_list

    def generate_params_to_learn(self, apop_intr, apop_extr):
        if apop_intr is not None:
            numb_intrs = apop_intr.param_idx.shape[0]
        else:
            numb_intrs = 0
        if apop_extr is not None:
            numb_extrs = apop_extr.param_idx.shape[0]
        else:
            numb_extrs = 0
        
        self.register_parameter(
                name="intr_param",
                param=nn.Parameter(torch.ones([numb_intrs, 3], device=self.device), requires_grad=False)) # f, cx, cy
        
        self.register_parameter(
                name="extr_param",
                param=nn.Parameter(torch.ones([numb_extrs, 6], device=self.device), requires_grad=False)) # se(3) to SE(3)

        return numb_intrs, numb_extrs

    def add_weights2params(self, intr, extr):
        if intr:
            intr_adj = self.add_weights2intr(self.apop_intr.img_h, self.apop_intr.img_w, adj=True)
        else:
            intr_adj = self.add_weights2intr(100, 100, adj=False) # fake intrinsics
        
        if extr:
            pose_adj = self.add_weights2pose(adj=True)
        else:
            pose_adj = self.add_weights2pose(adj=False)
     
        
        return intr_adj, pose_adj

    def add_weights2intr(self, img_h, img_w, adj=True):        
        intr_init = torch.tensor([[img_w, 0, img_w/2],
                                  [0, img_w, img_h/2],
                                  [0,       0,       1]], device=self.device).expand(self.intr_numb, 3, 3)
        intr_adj = intr_init.clone()
        if adj:
            intr_adj[:, 0, 0] = torch.abs(intr_init[:, 0, 0]*self.intr_param[:, 0].requires_grad_(True))
            intr_adj[:, 1, 1] = torch.abs(intr_init[:, 1, 1]*self.intr_param[:, 0].requires_grad_(True))
            intr_adj[:, 0, 2] = torch.abs(intr_init[:, 0, 2]*self.intr_param[:, 1].requires_grad_(True))
            intr_adj[:, 1, 2] = torch.abs(intr_init[:, 1, 2]*self.intr_param[:, 2].requires_grad_(True))    
        else:
            intr_adj[:, 0, 0] = torch.abs(intr_init[:, 0, 0]*self.intr_param[:, 0].requires_grad_(False))
            intr_adj[:, 1, 1] = torch.abs(intr_init[:, 1, 1]*self.intr_param[:, 0].requires_grad_(False))
            intr_adj[:, 0, 2] = torch.abs(intr_init[:, 0, 2]*self.intr_param[:, 0].requires_grad_(False))
            intr_adj[:, 1, 2] = torch.abs(intr_init[:, 1, 2]*self.intr_param[:, 0].requires_grad_(False))              
        return intr_adj

    def add_weights2pose(self, adj=True):
        if adj:
            weights_RT = self.se3_to_SE3(self.extr_param.requires_grad_(True))
        else:
            weights_RT = self.se3_to_SE3(self.extr_param.requires_grad_(False))        

        return weights_RT 

    def se3_to_SE3(self, tangent_vector):

        tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
        tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

        theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
        theta2 = theta**2
        theta3 = theta**3

        near_zero = theta < 1e-2
        non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
        theta_nz = torch.where(near_zero, non_zero, theta)
        theta2_nz = torch.where(near_zero, non_zero, theta2)
        theta3_nz = torch.where(near_zero, non_zero, theta3)

        # Compute the rotation
        sine = theta.sin()
        cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
        sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
        one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
        ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
        ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

        ret[:, 0, 0] += cosine.view(-1)
        ret[:, 1, 1] += cosine.view(-1)
        ret[:, 2, 2] += cosine.view(-1)
        temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
        ret[:, 0, 1] -= temp[:, 2]
        ret[:, 1, 0] += temp[:, 2]
        ret[:, 0, 2] += temp[:, 1]
        ret[:, 2, 0] -= temp[:, 1]
        ret[:, 1, 2] -= temp[:, 0]
        ret[:, 2, 1] += temp[:, 0]

        # Compute the translation
        sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
        one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
        theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

        ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
        ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
        ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
            tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
        )
        return ret

    @torch.no_grad()
    def show_3D_reproject_pts(self, images, pts_3d, pts_2d_gt):
        intr_adj, pose_adj = self.add_weights2params(self.apop_intr, self.apop_extr)
        if not self.opt_intr:
            intr_adj = self.intr_fix

        os.makedirs(self.reproj_2d_pts_save_pth, exist_ok=True)
        # pts_3d: [1, 100, 15, 3]
        # intr_adj: [100, 2]
        # pose_adj:
        reproj_pts = self.get_reproject_pixels(pts_3d.to(self.device), intr_adj, pose_adj)

        # 循环处理数据
        for ii, ori_img in enumerate(images):
            cur_img = ori_img.copy()
            # [15, 2]
            detect_pts = np.array(pts_2d_gt[ii].cpu()).astype(np.int32)
            cur_pd_pts = np.array(reproj_pts[ii].cpu()).astype(np.int32)
            for data in zip(detect_pts, cur_pd_pts):
                pt1, pt2 = data
                cv2.circle(cur_img, pt1, 2, [255,0,0], -1)
                cv2.circle(cur_img, pt2, 2, [0,255,0], -1)
    
            cv2.imwrite(os.path.join(Path(self.reproj_2d_pts_save_pth), Path("{}.png".format(ii))), cur_img)

    @torch.no_grad()  
    def show_RT_est_results(self):
        intr_adj, w2c_adj = self.add_weights2params(self.apop_intr, self.apop_extr)
        if not self.opt_intr:
            intr_adj = self.intr_fix
        os.makedirs(self.reproj_2d_pts_save_pth, exist_ok=True)
        # plt.ion()
        # plt.cla()
        color_pd = (0,0.6,0.7)
        clip_pose = w2c_adj[:, :3, :]
        # camera coord define as OpenCV
        self.draw_camera_shape(self.inverse_w2c(clip_pose), intr_adj, color_pd, cam_size=0.05)

        origin = [0, 0, 0]
        x_axis = [0.1, 0, 0]
        y_axis = [0, 0.1, 0]
        z_axis = [0, 0, 0.1]
        self.ax_3d.quiver(*origin, *x_axis, color='r', length=1.0, normalize=False)
        self.ax_3d.quiver(*origin, *y_axis, color='g', length=1.0, normalize=False)
        self.ax_3d.quiver(*origin, *z_axis, color='b', length=1.0, normalize=False)

        # file_path = os.path.join(Path(self.reproj_2d_pts_save_pth), Path("pose.png"))
        # plt.savefig(file_path)
        plt.show()

    def init_show_figure(self, use_2d=False):
        self.all_fig = plt.figure(figsize=(16,9))
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['mathtext.default'] = 'regular'

        if use_2d:
            self.ax_2d = self.all_fig.add_subplot(111)
        else:
            self.ax_3d = Axes3D(self.all_fig, auto_add_to_figure=False)
            self.all_fig.add_axes(self.ax_3d)
            self.ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            self.ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            self.ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                
            # if show_info:
            #     self.ax.set_xlabel("X Axis")
            #     self.ax.set_ylabel("Y Axis")
            #     self.ax.set_zlabel("Z Axis")
            #     self.ax.set_xlim(-3.5, 3.5)
            #     self.ax.set_ylim(-3.5, 3.5)
            #     self.ax.set_zlim(-1.5, 3.5)
            # else:
            #     self.ax.grid(False)
            #     self.ax.axis(False)
            
            plt.gca().set_box_aspect((1, 1, 1))

    def draw_camera_shape(self, extr_mat, intr_mat, color, cam_size=0.25):
        # extr_mat: [84, 3, 4]
        # intr_mat: [84, 3, 3]
        cam_line = cam_size
        focal = intr_mat[:,0,0]*cam_line/self.apop_extr.img_w
        cam_pts_1 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                                 -torch.ones_like(focal)*cam_line/2,
                                 focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_2 = torch.stack([-torch.ones_like(focal)*cam_line/2,
                                  torch.ones_like(focal)*cam_line/2,
                                 focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_3 = torch.stack([ torch.ones_like(focal)*cam_line/2,
                                  torch.ones_like(focal)*cam_line/2,
                                  focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_4 = torch.stack([ torch.ones_like(focal)*cam_line/2,
                                 -torch.ones_like(focal)*cam_line/2,
                                 focal], -1)[:,None,:].to(extr_mat.device)
        cam_pts_1 = cam_pts_1 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_2 = cam_pts_2 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_3 = cam_pts_3 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts_4 = cam_pts_4 @ extr_mat[:, :3, :3].transpose(-2,-1) + extr_mat[:, :3, 3][:,None,:]
        cam_pts = torch.cat([cam_pts_1, cam_pts_2, cam_pts_3, cam_pts_4, cam_pts_1], dim=-2)
        
        for i in range(4):
            # [84, 2, 3]
            cur_line_pts = torch.stack([cam_pts[:,i,:], cam_pts[:,i+1,:]], dim=-2).to('cpu')
            for each_cam in cur_line_pts:
                self.ax_3d.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=0.5)
        extr_T = extr_mat[:, :3, 3]
        
        for i in range(4):
            # [84, 2, 3]
            cur_line_pts = torch.stack([extr_T, cam_pts[:,i,:]], dim=-2).to('cpu')
            for each_cam in cur_line_pts:
                self.ax_3d.plot(each_cam[:,0],each_cam[:,1],each_cam[:,2],color=color,linewidth=0.5)

        extr_T = extr_T.to('cpu')

        self.ax_3d.scatter(extr_T[:,0],extr_T[:,1],extr_T[:,2],color=color,s=5)

    def inverse_w2c(self, c2w_mat):
        # print(c2w_mat.shape)
        ori_R = c2w_mat[:, :3, :3]
        ori_t = c2w_mat[:, :3, 3:]
        ori_R_T = ori_R.transpose(-1, -2)
        # print(ori_R_T.shape, ori_t.shape)
        T_new = -ori_R_T @ ori_t
        T_w2c = torch.eye(4, device=self.device)[None, ...].repeat(c2w_mat.shape[0], 1, 1)
        T_w2c[:, :3, :3] = ori_R_T
        T_w2c[:, :3, 3:] = T_new

        return T_w2c
    
    # save camera parameters
    def save_camera_parameters(self):
        data = {}
        for param_idx, name in zip(self.apop_extr.param_idx, self.apop_extr.name_list):
            data[name] = {"intrinsics": self.intr_adj.detach().cpu().to(torch.float32)[param_idx].tolist(),
                          "extrinsics_w2c": self.pose_adj.detach().cpu().to(torch.float32)[param_idx].tolist(),
                          "extrinsics_c2w": self.inverse_w2c(self.pose_adj).detach().cpu().to(torch.float32)[param_idx].tolist()}
        filename = os.path.join(Path(self.save_param_path), Path("cam_params.json"))
        # save as json
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)

        print(f"Parameters Save as {filename}".format(filename))


class Reprojection_Loss(nn.Module):
    def __init__(self, config, apop_intr, apop_extr):
        super(Reprojection_Loss, self).__init__()
        self.config = config
        self.apop_intr, self.apop_extr = apop_intr, apop_extr
        self.loss_l2 = nn.MSELoss(reduction='mean')
        # tensorboard
        self.global_step = 0
        self.img_h = self.apop_extr.img_h
        self.img_w = self.apop_extr.img_w

    def forward(self, loss_dict, gt_extr_pts, gt_intr_pts=None):
        self.global_step += 1

        if self.config["intrinsics"]:
            loss_intr = self.get_reproject_loss(loss_dict["intr"], gt_intr_pts)
        else:
            loss_intr = 0.0
        if self.config["extrinsics"]:
            loss_extr = self.get_reproject_loss(loss_dict["extr"], gt_extr_pts)
        else:
            loss_extr = 0.0

        final_loss = loss_intr + loss_extr

        return final_loss

    def get_reproject_loss(self, pd_pts, gt_pts):

        # pts format [x, y]        
        pd_pts_nx = pd_pts[..., 0]
        pd_pts_ny = pd_pts[..., 1]
        gt_pts_nx = gt_pts[..., 0]
        gt_pts_ny = gt_pts[..., 1]
 
        proj_loss_x = self.loss_l2(pd_pts_nx/self.img_w, gt_pts_nx/self.img_w)
        proj_loss_y = self.loss_l2(pd_pts_ny/self.img_h, gt_pts_ny/self.img_h)

        return proj_loss_x + proj_loss_y


if __name__ == "__main__":
    # config setting
    config = {"extrinsics_root": Path("data/table/images/rgbs_calib"), # images root
              "intrinsics_root": None, # images root
              "reproj_save_path": Path("outputs"), # visiualize image save path
              "param_save_path": Path("outputs"), # optimized parameters save path
              "tag_cube": "LAMPS",           # Apriltag type
              "tag_size": 0.20,              # meter level
              "intrinsics": False,           # estimate intrinsics
              "extrinsics": True,            # estimate extrinsics
              "intrinsics_path": "data/table/camera_params.json",
              "batch": 1,
              "total_step": 5000,
              "learning_rate": 0.1,
              "device": 'cuda',              
              }

    if config["intrinsics"]:
        apop_intrinsics = AprilTag_Operation(config, mode="intrinsics")
    else:
        apop_intrinsics = None

    if config["extrinsics"]:
        apop_extrinsics = AprilTag_Operation(config, mode="extrinsics")
    else:
        apop_extrinsics = None

    train_set = Data_set(apop_intrinsics, apop_extrinsics)

    reproject_3d_to_2d = Optimization_For_Reprojection(config, apop_intrinsics, apop_extrinsics)

    loss_func = Reprojection_Loss(config, apop_intrinsics, apop_extrinsics)
    
    optimizer = torch.optim.RAdam(reproject_3d_to_2d.parameters(), lr=config["learning_rate"], eps=1e-15)

    # 训练进度计数器
    cur_step = 0
    # training
    with tqdm(total = config["total_step"], desc='Optimization:',
              bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt} {postfix}]', ncols=150) as bar:
        running_loss = 0

        for step in range(config["total_step"]):
            optimizer.zero_grad()
            outputs = reproject_3d_to_2d(train_set.extr_w_coords.to(config["device"]))
            loss_final = loss_func.forward(outputs, train_set.extr_p_coords.to(config["device"]))
            loss_final.backward()
            optimizer.step()         
            running_loss += loss_final.item()
            ave_loss = running_loss/(step + 1)
            cur_step += 1
            bar.set_postfix_str('Loss:{:^7.9f}, LR:{:^7.9f}'.format(ave_loss, optimizer.param_groups[0]['lr']))
            bar.update()
            
            if step % 1000 == 0:
                reproject_3d_to_2d.show_3D_reproject_pts(train_set.extr_images, train_set.extr_w_coords, train_set.extr_p_coords)
    
        reproject_3d_to_2d.show_RT_est_results()
        reproject_3d_to_2d.save_camera_parameters()

