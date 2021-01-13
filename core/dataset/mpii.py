import cv2
import random
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from utils.transforms import get_affine_transform, affine_transform
from utils.transforms import normalize_coords
from utils.functions import debug_visualize

import torch
from torch.utils.data import Dataset

# loading MPII dataset


def prepare_data(cfg, tar_joint_num=16):
    cnt = 0
    data_images = []
    data_annots = []

    mat = io.loadmat(cfg.annot)

    for _, (anno, train_flag) in enumerate(  # all images
        zip(mat["RELEASE"]["annolist"][0, 0][0], mat["RELEASE"]["img_train"][0, 0][0])
    ):

        img_fn = anno["image"]["name"][0, 0][0]
        train_flag = int(train_flag)

        head_rect = []
        if "x1" in str(anno["annorect"].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno["annorect"]["x1"][0]],
                [y1[0, 0] for y1 in anno["annorect"]["y1"][0]],
                [x2[0, 0] for x2 in anno["annorect"]["x2"][0]],
                [y2[0, 0] for y2 in anno["annorect"]["y2"][0]],
            )
        else:
            head_rect = []  # TODO


        if "annopoints" in str(anno["annorect"].dtype):
            annopoints = anno["annorect"]["annopoints"][0]
            head_x1s = anno["annorect"]["x1"][0]
            head_y1s = anno["annorect"]["y1"][0]
            head_x2s = anno["annorect"]["x2"][0]
            head_y2s = anno["annorect"]["y2"][0]

        
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                annopoints, head_x1s, head_y1s, head_x2s, head_y2s
            ):

                if annopoint.size:
                    head_rect = [
                        float(head_x1[0, 0]),
                        float(head_y1[0, 0]),
                        float(head_x2[0, 0]),
                        float(head_y2[0, 0]),
                    ]

                    # joint coordinates
                    annopoint = annopoint["point"][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint["id"][0]]
                    x = [x[0, 0] for x in annopoint["x"][0]]
                    y = [y[0, 0] for y in annopoint["y"][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[int(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visibility list
                    if "is_visible" in str(annopoint.dtype):
                        vis = [
                            v[0] if v.size > 0 else [0]
                            for v in annopoint["is_visible"][0]
                        ]
                        vis = dict(
                            [
                                (k, int(v[0])) if len(v) > 0 else v
                                for k, v in zip(j_id, vis)
                            ]
                        )
                    else:
                        vis = None
                    
                    if len(joint_pos) >= tar_joint_num:
                        flag = True
                        # delete noise dataset
                        # for i in range(16):
                        #     for j in range(16):
                        #         if i == j: continue                        
                                
                        #         if joint_pos[i][0] == joint_pos[j][0] and joint_pos[i][1] == joint_pos[j][1]:
                        #             cnt += 1
                        #             flag = False
                        #             break
                            
                        #     if not flag: break
                        # if not flag: continue
                        

                        if joint_pos[0][0] == joint_pos[5][0] and joint_pos[0][1] == joint_pos[5][1]:
                            cnt += 1
                            continue
                        
                        
                        objpos = np.array([v[1] for v in sorted(joint_pos.items(), key=lambda x: x[0])]).reshape(16, 2)
                        xxyy = [
                            np.min(objpos[:, 0]),
                            np.max(objpos[:, 0]),
                            np.min(objpos[:, 1]),
                            np.max(objpos[:, 1]),
                        ]

                        w, h = abs(xxyy[1]-xxyy[0]), abs(xxyy[3] - xxyy[2])
                        bbox = [
                            (xxyy[0]+xxyy[1])/2.0,
                            (xxyy[2]+xxyy[3])/2.0,
                            w, h
                        ]

                        data = {
                            "filename": img_fn,
                            "train": train_flag,
                            "head_rect": head_rect,
                            "is_visible": vis,
                            "joint_pos": joint_pos,
                            "bbox": bbox,
                        }

                        if train_flag:
                            data_images.append(img_fn)
                            data_annots.append(data)
    print(f"Exclude {cnt} samples")
    return data_images, data_annots


class MPII_Dataset(Dataset):
    """
    0 - r ankle  |  1 - r knee  |  2 - r hip
    3 - l hip  |  4 - l knee  |  5 - l ankle

    6 - pelvis -> 골반
    7 - thorax -> 흉부
    8 - upper neck -> 목
    9 - head top -> 정수리

    10 - r wrist  |  11 - r elbow  |  12 - r shoulder
    13 - l shoulder  |  14 - l elbow  |  15 - l wrist
    """

    def __init__(self, cfg, data_pair, train):
        super(MPII_Dataset, self).__init__()

        self.cfg = cfg
        self.train = train
        self.data_image = data_pair[0]
        self.data_annot = data_pair[1]

    def __len__(self):
        return len(self.data_image)

    def __getitem__(self, index):
        i_name = self.data_image[index]
        i_meta = self.data_annot[index]


        # filename check
        meta_name = i_meta["filename"]
        assert meta_name == i_name


        # image load
        img_path = self.cfg.image / i_name
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # visibility and joints pose load
        meta_visible = sorted(i_meta["is_visible"].items(), key=lambda x: int(x[0]))
        meta_joints = sorted(i_meta["joint_pos"].items(), key=lambda x: int(x[0]))
        meta_xywh = i_meta["bbox"]
        meta_xyxy = [
            int(meta_xywh[0] - meta_xywh[2]*.5*1.2), 
            int(meta_xywh[1] - meta_xywh[3]*.5*1.1), 
            int(meta_xywh[0] + meta_xywh[2]*.5*1.3), 
            int(meta_xywh[1] + meta_xywh[3]*.5*1.2), 
        ]

        # meta_visible = np.array([v[1] for v in meta_visible if v[0] not in ["6", "7"]]).reshape(14, 1)
        # meta_joints = np.array([v[1] for v in meta_joints if v[0] not in [6, 7]]).reshape(14, 2)
        meta_visible = np.array([v[1] for v in meta_visible]).reshape(16, 1)
        meta_joints = np.array([v[1] for v in meta_joints]).reshape(16, 2)

        # debuging
        if self.cfg.debug:
            debug_visualize(self.cfg.debug_path, index, img, meta_visible, meta_joints, meta_xyxy, True, postfix="Original")


        if self.train:
            # lr flipping
            if self.cfg.flip:
                if np.random.random() <= 0.5:
                    img = img[:, ::-1, :]
                    meta_xywh[0] = img.shape[1] - 1 - meta_xywh[0]

                    meta_joints[:, 0] = img.shape[1] - 1 - meta_joints[:, 0]
                    for (q, w) in self.cfg.flip_pairs:
                        meta_joints_q, meta_joints_w = (
                            meta_joints[q, :].copy(),
                            meta_joints[w, :].copy(),
                        )
                        meta_joints[w, :], meta_joints[q, :] = meta_joints_q, meta_joints_w

                    # if self.cfg.debug:
                        # debug_visualize(self.cfg.debug_path, index, img, meta_visible, meta_joints, meta_xyxy, True, postfix="Flipped")

            # rotating and cropped
            if self.cfg.affine_transform:
                centre = np.array([
                    img.shape[1]/2.,
                    img.shape[0]/2.
                ])
                
                scale = np.array(img.shape[:2][::-1])
                rotation = 0
                
                if self.cfg.rotate:
                    if random.random() <= 0.6:
                        rotation = np.clip(np.random.randn()*self.cfg.rotate_factor, -self.cfg.rotate_factor, self.cfg.rotate_factor)
                
                trans = get_affine_transform(centre, scale, rotation, (img.shape[1], img.shape[0]))
                # cropped_img = cv2.warpAffine(img, trans, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
            
                for j in range(self.cfg.joints_num):
                    meta_joints[j, :2] = affine_transform(
                        meta_joints[j, :2], trans)

                # if self.cfg.debug:
                    # debug_visualize(self.cfg.debug_path, index, cropped_img, meta_visible, meta_joints, meta_xyxy, True, postfix="rotated")
            
            # normalize coordinates
            target_meta_joints, neck2toros_scaler, toros_centre = normalize_coords(meta_joints)
        
            visibility = np.array(
                [idx for idx, v in enumerate(meta_visible.reshape(-1, )) if v==1]
            )
            
            input_meta_joints = target_meta_joints.copy()

            
            if random.random() > 0.2:
                # random_mask_num = int(random.random() * (self.cfg.random_mask_num))
                random_mask_num = min(int(random.random() * (self.cfg.random_mask_num+1)), len(visibility))
                    
                if random_mask_num != 0:
                    random_mask = np.random.choice(visibility, random_mask_num, replace=False)
                    input_meta_joints[random_mask, :] = 0

            assert input_meta_joints.shape[0] == self.cfg.joints_num
            
            input_meta_joints = input_meta_joints.flatten()
            target_meta_joints = target_meta_joints.flatten()
            returns = {
                'inputs': torch.from_numpy(input_meta_joints.copy()).float(),
                'targets': torch.from_numpy(target_meta_joints.copy()).float(),
                'scaler': neck2toros_scaler,
                'centre': torch.from_numpy(toros_centre.copy()).float(),
                'bbox': torch.from_numpy(np.array(meta_xyxy).copy()).float(),
                'img_path': [str(img_path)],
            }
            
            return returns
        else:
            # normalize coordinates
            target_meta_joints, neck2toros_scaler, toros_centre = normalize_coords(meta_joints)
        
            visibility = np.array(
                [idx for idx, v in enumerate(meta_visible.reshape(-1, ))]
            )
            
            input_meta_joints = target_meta_joints.copy()
            random_mask_num = min(int(random.random() * (self.cfg.random_mask_num))+1, len(visibility))

            # if random.random() > 0.15:
                # random_mask_num = int(random.random() * (self.cfg.random_mask_num))
            random_mask_num = int(random.random() * 3) + 1
            if random_mask_num != 0:
                random_mask = np.random.choice(visibility, random_mask_num, replace=False)
                
                input_meta_joints[random_mask, :] = 0
                
            assert input_meta_joints.shape[0] == self.cfg.joints_num
            
            input_meta_joints = input_meta_joints.flatten()
            target_meta_joints = target_meta_joints.flatten()

            returns = {
                'inputs': torch.from_numpy(input_meta_joints.copy()).float(),
                'targets': torch.from_numpy(target_meta_joints.copy()).float(),
                'scaler': neck2toros_scaler,
                'centre': torch.from_numpy(toros_centre.copy()).float(),
                'bbox': torch.from_numpy(np.array(meta_xyxy).copy()).float(),
                'img_path': [str(img_path)],
            }
            
            return returns