import time
import logging
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummaryM import summary

from .baseLine import Base
from utils.functions import AverageMeter, valid_visualize
from utils.PID import PIDController


class VAE(Base):
    def __init__(self, cfg):
        super(VAE, self).__init__(cfg)

        # Build Encoder
        self._encoder_construct()

        # Build Decoder
        self._decoder_construct()

        # Build PID Controller
        self.C = self.cfg.C
        self.pid_controller = PIDController(cfg)
        
        # Model Summary
        model_summary, _ = summary(self, torch.zeros((2, self.cfg.input_dim)))
        logging.info(f"====Model Summary====\n{model_summary}\n")


    def _encoder_construct(self):
        modules = []
        if self.cfg.encode_hidden_dims is None:
            hidden_dims = [64, 128]
        else:
            hidden_dims = self.cfg.encode_hidden_dims


        # Build Encoder
        logging.info("Model Construction ==> Encoder Building...")
        for hidden_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(self.input_dims, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                )
            )
            self.input_dims = hidden_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(
            nn.Linear(self.input_dims, self.latent_dim),
            # nn.ReLU()
        )
        self.fc_var = nn.Sequential(
            nn.Linear(self.input_dims, self.latent_dim),
            # nn.ReLU()
        )
        logging.info("Model Construction ==> Done.\n")


    def _decoder_construct(self):
        logging.info("Model Construction ==> Decoder Building...")
        modules = []        
        if self.cfg.decode_hidden_dims is None:
            hidden_dims = [64, 16]
        else:
            hidden_dims = self.cfg.decode_hidden_dims
        
        self.decode_input = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.ReLU(),
        )

        for i in range(len(hidden_dims)):
            if i != len(hidden_dims)-1:
                modules.append(
                    nn.Sequential(
                        nn.Linear(self.input_dims, hidden_dims[i]), 
                        nn.BatchNorm1d(hidden_dims[i]),
                        nn.ReLU()
                    )
                )
            else:
                modules.append(
                    nn.Linear(self.input_dims, hidden_dims[i])
                )
            
            self.input_dims = hidden_dims[i]
        self.decoder = nn.Sequential(*modules)
        logging.info("Model Construction ==> Done.\n")      
        
        
    def encode(self, input):
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


    def decode(self, z):
        z = self.decode_input(z)
        return self.decoder(z)


    def sampling(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.sampling(mu, log_var)
        return  [self.decode(z), input, mu, log_var]
    
    
    def loss_function(self, preds, target, kld_weight=1.0, valid=False):
        reconstruction = preds[0]
        preds_mu = preds[2]
        preds_log_var = preds[3]

        # calc loss
        recons_loss = torch.sum(F.mse_loss(reconstruction, target, reduction='none'), dim=-1).mean()
        kld_loss = torch.mean(-0.5 * torch.sum(1 + preds_log_var - preds_mu ** 2 - preds_log_var.exp(), dim = -1))

        if self.cfg.control:
            if not valid:
                if (self.global_iter%self.cfg.period)==0:
                    self.C += self.cfg.C_step_val
                    
                if self.C > self.cfg.C_max:
                    self.C = self.cfg.C_max
                
                kld_weight, _ = self.pid_controller.pid(self.C, kld_loss.item())
                print(f"PID KL-Divergence Coefficient: {kld_weight}")
            # C = torch.clamp(self.C_max/self.cfg.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
            # loss = recons_loss + self.cfg.gamma*(kld_loss-C).abs()
            loss = recons_loss + kld_weight * kld_loss
        else:
            loss = recons_loss + kld_loss
        
        
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}
    
        
    def step(self, data, valid=False):
        if not valid: self.global_iter += 1

        inputs = data['inputs']
        target = data['targets']
        
        if self.cfg.device == "cuda":
            inputs = inputs.cuda()
            target = target.cuda()
        
        self.optim.zero_grad()
        preds = self.forward(inputs)
        losses = self.loss_function(preds, target, valid=valid)

        if not valid:
            losses['loss'].backward()
        self.optim.step()
        
        if not valid: return losses
        else: return preds, losses
        
        
    def train_on_epoch(self, epoch):
        self.train()
        self.set_loader(self.t_dset, self.cfg.train_batch, shuffle=True)
        self.C_max = torch.FloatTensor([self.cfg.C_max]).to(self.cfg.device)

        log_interval = min(len(self.loader)//10, self.cfg.log_interval)
        log_interval = max(log_interval, 1)
        
        str_time = time.time()
        logging.info(f'Train ==> Training...')
        
        total_loss = AverageMeter()
        kldiv_loss = AverageMeter()
        recon_loss = AverageMeter()
        batch_time = AverageMeter()
        for batch, data in enumerate(self.loader):
            batch_size = data['inputs'].size(0)
            losses = self.step(data)
            
            batch_time.update(time.time()-str_time)
            total_loss.update(losses['loss'].item())
            recon_loss.update(losses['Reconstruction_Loss'].item())
            kldiv_loss.update(losses['KLD'].item())
            

            if batch % log_interval == 0 or batch == len(self.loader) - 1:
                template = f'Train ==> Epoch [{str(epoch+1).zfill(len(str(self.cfg.epochs)))} | {str(self.cfg.epochs)}] \
->  Batch [{str(batch+1).zfill(len(str(len(self.loader))))} | {str(len(self.loader))}] \
->  Loss: {total_loss.avg:.6f} (Reconstruction -> {recon_loss.avg:12.6f}  |  KL-Div -> {kldiv_loss.avg:12.6f}) \
->  Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \
->  Speed: {batch_size/batch_time.val:.1f} samples/s \
->  Left: {batch_time.avg*(len(self.loader)-1-batch):.3f}s'
                    
                logging.info(template)
                
            str_time = time.time()
        logging.info("Train ==> Done.\n")


    def valid_on_epoch(self, epoch, log_interval=None):
        self.eval()
        self.set_loader(self.v_dset, self.cfg.valid_batch, shuffle=False)
        if log_interval is None:
            log_interval = min(len(self.loader)//10, self.cfg.log_interval)
            log_interval = max(log_interval, 1)
        
        count = 1
        str_time = time.time()
        logging.info(f'Valid ==> Validation...')
        
        total_loss = AverageMeter()
        recon_loss = AverageMeter()
        kldiv_loss = AverageMeter()
        batch_time = AverageMeter()

        with torch.no_grad():
            for batch, data in enumerate(self.loader):
                batch_size = data['inputs'].size(0)
                preds, losses = self.step(data, True)
                
                batch_time.update(time.time()-str_time)
                total_loss.update(losses['loss'].item())
                recon_loss.update(losses['Reconstruction_Loss'].item())
                kldiv_loss.update(losses['KLD'].item())
                
                
                if self.cfg.valid_visualize and batch % log_interval == 0:
                    valid_visualize(epoch, count, data, preds, self.cfg.valid_visualization, self.cfg.valid_visualize_num)
                    count += batch_size


                if batch % log_interval == 0 or batch == len(self.loader) - 1:
                    template = f'Valid ==> Epoch [{str(epoch+1).zfill(len(str(self.cfg.epochs)))} | {str(self.cfg.epochs)}] \
    ->  Batch [{str(batch+1).zfill(len(str(len(self.loader))))} | {str(len(self.loader))}] \
    ->  Loss: {total_loss.avg:.6f} (Reconstruction -> {recon_loss.avg:12.6f}  |  KL-Div -> {kldiv_loss.avg:12.6f}) \
    ->  Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s) \
    ->  Speed: {batch_size/batch_time.val:.1f} samples/s \
    ->  Left: {batch_time.avg*(len(self.loader)-1-batch):.3f}s'

                    logging.info(template)
                str_time = time.time()
        logging.info("Valid ==> Done.\n")
        return total_loss, recon_loss, kldiv_loss
    
    
    def save(self, epoch, t_loss, r_loss, kld_loss, best_loss):
        try:
            state_dict = self.module.state_dict()
        except Exception as e:
            state_dict = self.state_dict()
            
        status = {
            'epoch': epoch,
            't_loss': t_loss.avg,
            'state_dict': state_dict,
            'optim': self.optim.state_dict(),
            
        }
        
        filename = str(status['epoch']).zfill(3) + "_.pth.tar"
        torch.save(status, osp.join(self.cfg.checkpoint, filename))
        
        logging.info(
            f'Save Best Model ==> Epoch:{epoch+1}  |  Best Loss: {best_loss:.6f}  ----->  {t_loss.avg:.6f}'
        )
        logging.info(
            f'Save Best Model ==> Epoch:{epoch+1}  |  [Best Loss: {t_loss.avg:.6f}  |  Reconstruction Loss: {r_loss.avg:.6f}  |  KL-Divergence Loss: {kld_loss.avg:.6f}'
        )


    def load(self, ckpt):
        if osp.exists(ckpt):
            save_dict = torch.load(ckpt)

            epoch = save_dict["epoch"]
            state_dict =  save_dict["state_dict"]
            optim_dict =  save_dict["optim"]

            self.optim.load_state_dict(optim_dict)
            try:
                self.module.load_state_dict(state_dict)
            except AttributeError as e:
                self.load_state_dict(state_dict)
            self.cfg.start_epoch = epoch

            logging.info(f"{ckpt} model loaded.")
            logging.info(f"Epoch re-start at {epoch}.\n")
        else:
            # raise FileNotFoundError("File doesn't exsist. Please check the directory.")
            logging.info("File doesn't exsist. Please check the directory.")
            logging.info("Start training with initailized model.\n")
        


if __name__ == "__main__":
    from pathlib import Path
    from torchsummaryM import summary
    
    class config:
        def __init__(self):
            # model
            self.latent_dim = 16
            self.encode_hidden_dims = [64, 128]
            self.decode_hidden_dims = [64, 16]
            

    cfg = config()
    model = VAE(cfg)
    summary(model, torch.zeros((2, 16)))