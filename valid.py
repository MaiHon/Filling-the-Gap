import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import cfg
from core.dataset.mpii import MPII_Dataset, prepare_data
from core.models.model import VAE

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
    
    
    for epoch in range(cfg.start_epoch, cfg.epochs):
        t_loss, r_loss, kld_loss = model.valid_on_epoch(epoch, log_interval=1)


if __name__ == "__main__":
    main()