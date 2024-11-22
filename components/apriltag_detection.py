import torch
import apriltag
import cv2
import os
import random

import numpy as np

from pathlib import Path

class AprilTag_Operation(object):
    def __init__(self, config, mode="extrinsics"):
        if mode == "extrinsics":
            self.path_root = config["extrinsics_root"]
        else:
            self.path_root = config["intrinsics_root"]
        # size in real world
        self.tag_size = config["tag_size"]
        # load images
        self.load_imgs, self.numb_imgs, self.name_imgs, self.img_h, self.img_w, self.rgb_imgs = self.load_images(self.path_root)
        # info detection
        self.tags_in_train_images = self.apriltag_detection(self.load_imgs, self.numb_imgs, self.name_imgs, mode)
        # generate calibration points in world space
        self.tags_world_coords = self.apriltag_gt_pts_real_world(self.tag_size)
        # generate training data
        self.pixel_coords, self.world_coords, self.param_idx, self.name_list = self.get_apriltag_pts_data(self.tags_in_train_images, self.tags_world_coords)

    # load images as np
    def load_images(self, path_root):
        img_names = os.listdir(path_root)
        img_names.sort()
        img_list = []
        rgb_img_list = []
        for cur_name in img_names:
            cur_img_pth = os.path.join(Path(path_root), Path(cur_name))
            img = cv2.imread(cur_img_pth, 0) # gray mode
            rgb_img = cv2.imread(cur_img_pth) # rgb mode
            img = cv2.normalize(img, None, 0, 250, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            img_list += [img]
            rgb_img_list += [rgb_img]
        img_numb = len(img_list)
        print("Including {} images from folder".format(img_numb))

        return img_list, img_numb, img_names, img.shape[0], img.shape[1], rgb_img_list

    # apriltag detection
    def apriltag_detection(self, images_including_apriltags, img_numbs, img_names, mode):
        ignore_names = []
        all_tag_info = {}
        all_id_info = {0:{'id':[],'pts':[]},\
                       1:{'id':[],'pts':[]},\
                       2:{'id':[],'pts':[]},\
                       3:{'id':[],'pts':[]},\
                       4:{'id':[],'pts':[]},\
                       5:{'id':[],'pts':[]}}
        detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        param_idx = 0
        detect_flag = 0

        for img_idx, cur_img in enumerate(images_including_apriltags):
            tags = detector.detect(cur_img)
            cur_name = img_names[img_idx]

            if len(tags) == 0:
                print("Apriltags in {} are not detected !!".format(cur_name))
                print("Ignore image {}".format(cur_name))
                ignore_names += [cur_name]
                continue

            # intrinsics must including more than 2 Apriltags in each images
            if (mode == "intrinsics") and (len(tags) < 2):
                print("Apriltags in {} needs more than 2 !!".format(cur_name))
                print("Ignore image {}".format(cur_name))
                ignore_names += [cur_name]
                continue

            tag_ids = []
            tag_pts = []
            for tag in tags:
                tag_id = tag.tag_id
                center_p = tag.center  # format:[x, y], also (w, h)
                corner_p = tag.corners # order:[lt, rt, rb， lb] 
                points_tag = np.concatenate([center_p.reshape([1, -1]), corner_p], 0)
                tag_pts += [points_tag]
                tag_ids += [tag_id]
                all_id_info[tag_id]['id'] += [param_idx]
                all_id_info[tag_id]['pts'] += [points_tag]
                
            # 保存当前图片中的检测信息
            all_tag_info[param_idx] = [tag_ids, tag_pts, cur_name]
            
            param_idx += 1
            detect_flag += 1

        if detect_flag == img_numbs:
            print("All images include calibration points....")
        else:
            print("Lacking of {} images !!".format(img_numbs - detect_flag))
            for ig_name in ignore_names:
                print("Lacking: ".format(ig_name))

        return all_tag_info

    # generate traing data
    def get_apriltag_pts_data(self, apriltag_dict, tag_world_pts):

        pixel_pts = []
        world_pts = []
        param_idx = []
        img_names = []

        for img_id, info in apriltag_dict.items():
            pts_list = []
            wpts_list = []
            cur_name = info[2]
            # expend to 3 apriltag faces
            for idx in range(len(info[0])):
                tag_id = info[0][idx]
                tag_pt = info[1][idx]
                tag_wpts = tag_world_pts[tag_id]
                pts_list += [torch.from_numpy(tag_pt).to(torch.float32)]
                wpts_list += [tag_wpts]
            # only 1 apriltag in image
            if len(pts_list) == 1:
                pts_list += [pts_list[0], pts_list[0]]
                wpts_list += [wpts_list[0], wpts_list[0]]
            # only 2 apriltags in image
            if len(pts_list) == 2:
                choice_id = random.randint(0, len(pts_list)-1)
                pts_list += [pts_list[choice_id]]
                wpts_list += [wpts_list[choice_id]]    
            
            # save infomation
            pixel_pts += [torch.cat(pts_list, 0)]
            world_pts += [torch.cat(wpts_list, 0)]
            param_idx  += [torch.tensor(img_id)]
            img_names += [cur_name]

        pixel_pts = torch.stack(pixel_pts, 0)
        world_pts = torch.stack(world_pts, 0)
        param_idx = torch.stack(param_idx, 0)

        return pixel_pts, world_pts, param_idx, img_names
        
    # generate world points for real world calibration cube
    def apriltag_gt_pts_real_world(self, tag_size):
        cube_half = tag_size/2
        tag_half = tag_size*0.8/2
        world_tag_pts = {0:[[0.0,       -cube_half,       0.0],
                            [-tag_half, -cube_half,  tag_half],
                            [ tag_half, -cube_half,  tag_half],
                            [ tag_half, -cube_half, -tag_half],
                            [-tag_half, -cube_half, -tag_half]],
                         1:[[  cube_half,      0.0,       0.0],
                            [  cube_half,-tag_half,  tag_half],
                            [  cube_half, tag_half,  tag_half],
                            [  cube_half, tag_half, -tag_half],
                            [  cube_half,-tag_half, -tag_half]],
                         2:[[  0.0,      cube_half,       0.0],
                            [ tag_half,  cube_half,  tag_half],
                            [-tag_half,  cube_half,  tag_half],
                            [-tag_half,  cube_half, -tag_half],
                            [ tag_half,  cube_half, -tag_half]],
                         3:[[ -cube_half,      0.0,       0.0],
                            [ -cube_half, tag_half,  tag_half],
                            [ -cube_half,-tag_half,  tag_half],
                            [ -cube_half,-tag_half, -tag_half],
                            [ -cube_half, tag_half, -tag_half]],
                         4:[[  0.0,            0.0,  cube_half],
                            [-tag_half,  -tag_half,  cube_half],
                            [-tag_half,   tag_half,  cube_half],
                            [ tag_half,   tag_half,  cube_half],
                            [ tag_half,  -tag_half,  cube_half]],
                         5:[[  0.0,            0.0, -cube_half],
                            [ tag_half,  -tag_half, -cube_half],
                            [ tag_half,   tag_half, -cube_half],
                            [-tag_half,   tag_half, -cube_half],
                            [-tag_half,  -tag_half, -cube_half]]}
        world_tag_pts_tensor = {}
        for key in world_tag_pts:
            world_tag_pts_tensor[key] = torch.tensor(world_tag_pts[key])
        return world_tag_pts_tensor

