import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from core.dataset.mpii import MPII_Dataset, prepare_data
from sklearn.model_selection  import train_test_split


def main():
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--m_type', default="vanila", type=str, help='Model types  [Vanila|Control]')
    
    args = parser.parse_args()
    if args.m_type == "vanila":
        from configs.vanila_config import cfg
        from core.models.vanila_vae import VAE
    elif args.m_type == "control":
        from configs.control_config import cfg
        from core.models.control_vae import VAE
    elif args.m_type == "ladder":
        from configs.ladder_config import cfg
        from core.models.ladder_vae import VAE
    
    
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
    
    print("Train dataset:", len(t_dset))
    print("Valid dataset:", len(v_dset))

    model = VAE(cfg)
    model.set_dsets((t_dset, v_dset))
    model.set_optim()
    # model.set_sched()
    model.gpu_check()

    if cfg.load_weights:
        model.load(cfg.weight_path)
    
    
    best_loss = float("INF")
    for epoch in range(cfg.start_epoch, cfg.epochs):
        # train
        model.train_on_epoch(epoch)
        
        # valid
        if epoch % cfg.valid_interval == 0:
            t_loss, r_loss, kld_loss = model.valid_on_epoch(epoch)

            if best_loss >= t_loss.avg:
                model.save(epoch, t_loss, r_loss, kld_loss, best_loss)
                best_loss = t_loss.avg


if __name__ == "__main__":
    main()