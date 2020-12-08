import time
import logging
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummaryM import summary


class Base(nn.Module):
    def __init__(self, cfg):
        super(Base, self).__init__()
        
        # logging setting
        logging.basicConfig(filename=cfg.log_dir,
                            filemode='w',
                            level=logging.INFO, 
                            format='%(asctime)s  |  %(message)s')        
        
        self.cfg = cfg
        self.global_iter = 0
        self.st_epoch = cfg.start_epoch
        self.latent_dim = cfg.latent_dim
        self.input_dims = self.cfg.input_dim


    def _encoder_construct(self):
        NotImplemented


    def _decoder_construct(self):
        NotImplemented


    def set_loader(self, dset, batch, shuffle):
        self.loader = DataLoader(
            dset, 
            batch_size=batch,
            shuffle=shuffle,
        )
        
    
    def set_sched(self):
        self.sched = optim.lr_scheduler.CyclicLR(
            self.optim, self.cfg.lr, self.cfg.lr*10, step_size_up=2000,
            cycle_momentum=False)


    def set_dsets(self, d_sets):
        logging.info("Model Construction ==> Dataset Setting...")
        
        self.t_dset = d_sets[0]
        self.v_dset = d_sets[1]
        
        logging.info("Model Construction ==> Done.\n")


    def set_optim(self):
        logging.info("Model Construction ==> Optimizer Setting...")
        
        if self.cfg.optim == "Adam":
            self.optim = optim.Adam(params=self.parameters(), 
                                    lr=self.cfg.lr)
        
        logging.info("Model Construction ==> Done.\n")


    def gpu_check(self):
        if self.cfg.device == "cuda":
            logging.info('Model Construction ==> Model on GPU.')
            self = self.cuda()
        else:
            logging.info('Model Construction ==> Model on CPU.')
            self = self.cpu()
        
        if torch.cuda.device_count() > 1:
            logging.info('Model Construction ==> Use Multi GPUs.')
            logging.info(f'Model Construction ==> Model on {torch.cuda.device_count()} GPUs.\n')
            return nn.DataParallel(self)
        else:
            logging.info('Model Construction ==> Use Single GPU.\n')
            return self            
        
        
    def encode(self):
        NotImplemented


    def decode(self):
        NotImplemented


    def sampling(self):
        NotImplemented


    def forward(self):
        NotImplemented
    
    
    def loss_function(self):
        NotImplemented
    
    
    def step(self, data, valid=False):
        NotImplemented
        
        
    def train_on_epoch(self):
        NotImplemented


    def valid_on_epoch(self):
        NotImplemented
    

    def save(self):
        NotImplemented


    def load(self):
        NotImplemented