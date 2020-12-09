import os
import os.path as osp
import cv2
import numpy as np
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        
        
def visualize_joints(img, kps, kp_thresh=0.4, alpha=0.6):
        """
        0 - r ankle  |  1 - r knee  |  2 - r hip
        3 - l hip  |  4 - l knee  |  5 - l ankle

        6 - pelvis -> 골반
        7 - thorax -> 흉부
        8 - upper neck -> 목
        9 - head top -> 대가리

        10 - r wrist  |  11 - r elbow  |  12 - r shoulder
        13 - l shoulder  |  14 - l elbow  |  15 - l wrist
        """
        kps_lines = [
            (5, 4),
            (4, 3),
            (2, 1),
            (1, 0),
            (15, 14),
            (14, 13),
            (12, 11),
            (11, 10),
            (6, 2),
            (6, 3),
            (6, 7),
            (7, 8),
            (8, 9),
            (7, 12),
            (7, 13),
        ]

        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        kp_mask = np.copy(img)

        # todo: do not need for mpii
        # Draw mid shoulder / mid hip first for better visualization.
        # mid_shoulder = (
        #     kps[12, :2] +
        #     kps[13, :2]) / 2.0
        # sc_mid_shoulder = np.minimum(
        #     kps[12, 2],
        #     kps[13, 2])
        # mid_hip = (
        #     kps[2, :2] +
        #     kps[3, :2]) / 2.0
        # sc_mid_hip = np.minimum(
        #     kps[2, 2],
        #     kps[3, 2])

        # head_top = 9
        # if sc_mid_shoulder > kp_thresh and kps[head_top, 2] > kp_thresh:
        #     cv2.line(
        #         kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(
        #             kps[head_top, :2].astype(np.int32)),
        #         color=colors[len(kps_lines)], thickness=2, lineType=cv2.LINE_AA)
        # if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        #     cv2.line(
        #         kp_mask, tuple(mid_shoulder.astype(np.int32)), tuple(
        #             mid_hip.astype(np.int32)),
        #         color=colors[len(kps_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

        # Draw the keypoints.
        for l in range(len(kps_lines)):
            i1 = kps_lines[l][0]
            i2 = kps_lines[l][1]
            p1 = kps[i1, 0].astype(np.int32), kps[i1, 1].astype(np.int32)
            p2 = kps[i2, 0].astype(np.int32), kps[i2, 1].astype(np.int32)
            if kps[i1, 2] > kp_thresh and kps[i2, 2] > kp_thresh:
                cv2.line(
                    kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA
                )
            if kps[i1, 2] > kp_thresh:
                cv2.circle(
                    kp_mask,
                    p1,
                    radius=3,
                    color=colors[l],
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            if kps[i2, 2] > kp_thresh:
                cv2.circle(
                    kp_mask,
                    p2,
                    radius=3,
                    color=colors[l],
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        # Blend the keypoints.
        return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def debug_visualize(debug_path, index, img, meta_visible, meta_joints, meta_bbox=None, with_visible=False, postfix=None):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if with_visible:
        meta_visible = np.array([1 for _ in range(len(meta_visible))]).reshape(
            16, 1
        )

    img_w_joitns = visualize_joints(
        img_bgr, np.concatenate([meta_joints, meta_visible], axis=-1)
    )

    if meta_bbox is not None:
        img_bgr = cv2.rectangle(img_w_joitns, 
                            tuple(meta_bbox[:2]),
                            tuple(meta_bbox[2:]), 
                            color=(0, 0, 254),
                            thickness=2, 
                            lineType=cv2.LINE_AA)

    if postfix is None:
        cv2.imwrite(str(debug_path / f"{index}_visualized.jpg"), img_w_joitns)
    else:
        cv2.imwrite(str(debug_path / f"{index}_{postfix}.jpg"), img_w_joitns)

    
def valid_visualize(epoch, count, data, preds, save_path, visualize_num=None):
    img_paths = data['img_path'][0]
    scaler    = data['scaler'].detach().cpu().numpy()
    centre    = data['centre'].detach().cpu().numpy()
    bboxes    = data['bbox'].detach().cpu().numpy()
    # masks     = data['mask'].detach().cpu().numpy()

    preds     = preds[0].detach().cpu().numpy().reshape(-1, 16, 2)
    inputs    = data['inputs'].detach().cpu().numpy().reshape(-1, 16, 2)
    targets   = data['targets'].detach().cpu().numpy().reshape(-1, 16, 2)

    if visualize_num is not None:
        candis = np.random.choice(range(targets.shape[0]), visualize_num, replace=False)
    else:
        candis = range(len(preds))

    for candi in candis:
        img_bgr = cv2.imread(str(img_paths[candi]))
        
        input_visibility = np.array([
            1 if i.all() else 0 for i in inputs[candi]
        ]).reshape(16, 1)
        
        pred = preds[candi] * scaler[candi] + centre[candi]
        input = inputs[candi] * scaler[candi] + centre[candi]
        target = targets[candi] * scaler[candi] + centre[candi]
        visibility = np.array([1 for _ in range(len(pred))]).reshape(16, 1)


        # draw joints
        img_w_pred = visualize_joints(
            img_bgr, np.concatenate([pred, visibility], axis=-1)
        )

        img_w_input = visualize_joints(
            img_bgr, np.concatenate([input, input_visibility], axis=-1)
        )
        
        img_w_orig = visualize_joints(
            img_bgr, np.concatenate([target, visibility], axis=-1)
        )

        # draw bbox
        img_w_pred = cv2.rectangle(img_w_pred, 
                            tuple(bboxes[candi][:2]),
                            tuple(bboxes[candi][2:]), 
                            color=(0, 0, 254),
                            thickness=2, 
                            lineType=cv2.LINE_AA)

        img_w_input = cv2.rectangle(img_w_input, 
                            tuple(bboxes[candi][:2]),
                            tuple(bboxes[candi][2:]), 
                            color=(0, 0, 254),
                            thickness=2, 
                            lineType=cv2.LINE_AA)
        
        img_w_orig = cv2.rectangle(img_w_orig, 
                            tuple(bboxes[candi][:2]),
                            tuple(bboxes[candi][2:]), 
                            color=(0, 0, 254),
                            thickness=2, 
                            lineType=cv2.LINE_AA)
        
        # save
        path = str(save_path / f"{epoch}" / "pred")
        if not osp.exists(path):
            os.makedirs(path, exist_ok=True)
        cv2.imwrite(osp.join(path, f"{count}_pred.jpg"), img_w_pred)

        path = str(save_path / f"{epoch}" / "input")
        if not osp.exists(path):
            os.makedirs(path, exist_ok=True)
        cv2.imwrite(osp.join(path, f"{count}_input.jpg"), img_w_input)

        path = str(save_path / f"{epoch}" / "orig")
        if not osp.exists(path):
            os.makedirs(path, exist_ok=True)
        cv2.imwrite(osp.join(path, f"{count}_orig.jpg"), img_w_orig)

        count += 1