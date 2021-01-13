import os
import os.path as osp
import cv2
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from configs.vanila_config import cfg
from core.dataset.mpii import MPII_Dataset, prepare_data
from core.models.vanila_vae import VAE
from core.utils.functions import visualize_joints

from sklearn.model_selection  import train_test_split


def main():
    random_seed = 2020
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    cudnn.benchmark = cfg.cudnn_benchmark
    torch.backends.cudnn.deterministic = cfg.cudnn_deterministic
    torch.backends.cudnn.enabled = cfg.cudnn_enable


    total_image, total_annot = prepare_data(cfg)
    train_image, valid_image, train_annot, valid_annot = train_test_split(total_image, total_annot, test_size=0.15, random_state=random_seed)

    t_dset = MPII_Dataset(cfg, (train_image, train_annot), train=True)
    v_dset = MPII_Dataset(cfg, (valid_image, valid_annot), train=False)

    model = VAE(cfg)
    model.set_dsets((t_dset, v_dset))
    model.set_optim()
    model.gpu_check()

    if cfg.load_weights:
        model.load(cfg.weight_path)
        print("model loaded.")
    
    count = 1
    model.eval()
    model.set_loader(model.t_dset, model.cfg.valid_batch, shuffle=False)
    for idx, data in enumerate(model.loader):
        input = data["inputs"].float()
        target = data["targets"].float()
        
        if model.cfg.device == "cuda":
            input = input.cuda()
            target = target.cuda()
        
        # with torch.no_grad():
        #     preds = model.forward(input)[0]
            
        # preds = preds.detach().cpu().numpy()
        # input = input.detach().cpu().numpy()
        
        # preds = np.where(input==0, preds, input)
        
        img_paths = data['img_path'][0]
        scaler    = data['scaler'].detach().cpu().numpy()
        centre    = data['centre'].detach().cpu().numpy()
        bboxes    = data['bbox'].detach().cpu().numpy()

        # preds     = preds.reshape(-1, 16, 2)
        # inputs    = input.reshape(-1, 16, 2)
        targets   = data['targets'].detach().cpu().numpy().reshape(-1, 16, 2)

    
        candis = range(len(targets))

        for candi in candis:
            img_bgr = cv2.imread(str(img_paths[candi]))
            
            # input_visibility = np.array([
            #     1 if i.all() else 0 for i in inputs[candi]
            # ]).reshape(16, 1)
        
            
            # pred = preds[candi] * scaler[candi] + centre[candi]
            # input = inputs[candi] * scaler[candi] + centre[candi]
            target = targets[candi] * scaler[candi] + centre[candi]
            visibility = np.array([1 for _ in range(len(target))]).reshape(16, 1)


            # draw joints
            # img_w_pred = visualize_joints(
            #     img_bgr, np.concatenate([pred, visibility], axis=-1)
            # )

            # img_w_input = visualize_joints(
            #     img_bgr, np.concatenate([input, input_visibility], axis=-1)
            # )
            
            img_w_orig = visualize_joints(
                img_bgr, np.concatenate([target, visibility], axis=-1)
            )

            # draw bbox
            # img_w_pred = cv2.rectangle(img_w_pred, 
            #                     tuple(bboxes[candi][:2]),
            #                     tuple(bboxes[candi][2:]), 
            #                     color=(0, 0, 254),
            #                     thickness=2, 
            #                     lineType=cv2.LINE_AA)

            # img_w_input = cv2.rectangle(img_w_input, 
            #                     tuple(bboxes[candi][:2]),
            #                     tuple(bboxes[candi][2:]), 
            #                     color=(0, 0, 254),
            #                     thickness=2, 
            #                     lineType=cv2.LINE_AA)
            
            img_w_orig = cv2.rectangle(img_w_orig, 
                                tuple(bboxes[candi][:2]),
                                tuple(bboxes[candi][2:]), 
                                color=(0, 0, 254),
                                thickness=2, 
                                lineType=cv2.LINE_AA)
            
            # save
            folder = "train"
            
            # path = str(model.cfg.root / folder / "pred")
            # if not osp.exists(path):
            #     os.makedirs(path, exist_ok=True)
            # cv2.imwrite(osp.join(path, f"{count}_pred.jpg"), img_w_pred)

            # path = str(model.cfg.root / folder / "input")
            # if not osp.exists(path):
            #     os.makedirs(path, exist_ok=True)
            # cv2.imwrite(osp.join(path, f"{count}_input.jpg"), img_w_input)

            path = str(model.cfg.root / folder  / "orig")
            if not osp.exists(path):
                os.makedirs(path, exist_ok=True)
            cv2.imwrite(osp.join(path, f"{count}_orig.jpg"), img_w_orig)

            count += 1


if __name__ == "__main__":
    main()