import time
import logging
import os.path as osp
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummaryM import summary

from .baseLine import Base
from utils.functions import AverageMeter, valid_visualize

class MLP(nn.Module):
    def __init__(self, ins, outs, latent):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(ins, outs),
            nn.ReLU(),
            nn.Linear(outs, outs),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(outs, latent)
        self.var = nn.Linear(outs, latent)
        
    def forward(self, x):
        mlp = self.mlp(x)
        
        mu, var = self.mu(mlp), self.var(mlp)
        # var = F.softplus(var) + 1e-6
        return mlp, mu, var


class VAE(Base):
    def __init__(self, cfg):
        super(VAE, self).__init__(cfg)

        # Build Encoder
        self._encoder_construct()

        # Build Decoder
        self._decoder_construct()

        # Model Summary
        model_summary, _ = summary(self, torch.zeros((2, self.cfg.input_dim)))
        logging.info(f"====Model Summary====\n{model_summary}\n")


    def _encoder_construct(self):
        logging.info("Model Construction ==> Encoder Building...")
        self.encoder_modules = nn.ModuleList()
        
        for i in range(1, self.cfg.l_layers+1):
            self.encoder_modules.append(
                MLP(self.input_dims, self.cfg.encode_hidden_dims[i-1], self.latent_dim)
            )
            
            self.input_dims = self.cfg.encode_hidden_dims[i-1]
        logging.info("Model Construction ==> Done.\n")


    def _decoder_construct(self):
        logging.info("Model Construction ==> Decoder Building...")

        self.decoder_modules = nn.ModuleList()
        for i in range(0, self.cfg.l_layers-1):
            self.decoder_modules.append(MLP(self.latent_dim, self.cfg.decode_hidden_dims[i], self.latent_dim))
            self.input_dims = self.cfg.decode_hidden_dims[i]
            
        self.decoder_modules.append(
            nn.Sequential(
                nn.Linear(self.latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, self.cfg.joints_num*2),
            )    
        )
        

        logging.info("Model Construction ==> Done.\n")      
        
        
    def encode(self, input):
        encode = input
        encoder_result = []
        for e in self.encoder_modules:
            encode, mu, var = e(encode)
            encoder_result.append([encode, mu, var])

        return_val = [0, 1]
        for idx, (encode, decode) in enumerate(zip(encoder_result[-1:0:-1], self.decoder_modules)):
            mu, var = encode[1:]
            z = self.sampling(mu, var)
            if idx == 0:
                return_val = [mu, var]
            
            _, mu_decode, var_decode = decode(z)
            
            prec_encode = encoder_result[idx][-1] ** (-1)
            prec_decode = var_decode ** (-1) 
            
            mu_encode = (mu * prec_encode + mu_decode * prec_decode) / (prec_encode + prec_decode)
            var_encode = (prec_encode + prec_decode) ** (-1)
            
            encoder_result[idx][1] = mu_encode
            encoder_result[idx][2] = var_encode

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        return return_val, encoder_result


    def decode(self, init_val, encoder_result):
        decoder_result = []
        
        val = init_val
        for idx, decode in enumerate(self.decoder_modules):
            z = self.sampling(*val)
            
            if idx < len(self.decoder_modules)-1:
                recon, mu, var = decode(z)
                decoder_result.append([mu, var])
                val = [mu, var]
            else:
                recon = decode(z)
        
        return recon, decoder_result


    def sampling(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def forward(self, input, **kwargs):
        init_val, encoder_result = self.encode(input)
        recon, decoder_result = self.decode(init_val, encoder_result)
        return recon, encoder_result, decoder_result
    
        
    def loss_function(self, preds, target, kld_weight=1.0):
        recon = preds[0]
        encoder_preds = preds[1][-1::-1]
        decoder_preds = preds[2]
        
        recons_loss = torch.sum(F.mse_loss(recon, target, reduction='none'), dim=-1).mean()
        kld_loss = torch.mean(
            -0.5 * torch.sum(
                1 + encoder_preds[0][2] - encoder_preds[0][1] ** 2 - encoder_preds[0][2].exp(), dim = -1
                )
            )
        
        
        # for i in range(len(decoder_preds)):
        #     kld_loss += torch.mean(
        #         -0.5 * torch.sum(
        #             decoder_preds[i][0] + encoder_preds[0][2] \
        #                 - (encoder_preds[0][1]-decoder_preds[i][1]) ** 2 - encoder_preds[0][2].exp(), dim = -1
        #         )
        #     )
        
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}
    
        
    def step(self, data, valid=False):
        self.global_iter += 1

        inputs = data['inputs']
        target = data['targets']
        
        if self.cfg.device == "cuda":
            inputs = inputs.cuda()
            target = target.cuda()
        
        self.optim.zero_grad()
        preds = self.forward(inputs)
        losses = self.loss_function(preds, target)

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