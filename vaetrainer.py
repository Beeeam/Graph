import os
import yaml
import pickle
import shutil
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW, Adam, SGD
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from rdkit import Chem
from transformers import get_scheduler
from utils import vae_loss, load_vocab_from_pickle, KLAnnealer, mask_predict_vae_loss
from data import MoleculeLoaderWrapper
from vae import GraphEncoder, SMILESDecoder, VAE

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

class VAE_trainer(object):
    def __init__(self, config, dataset, vocab):
        self.config =config
        self.dataset = dataset
        self.vocab = vocab

        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_'  + '_'
        log_dir = os.path.join('ckpt/vae', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def _get_device(self):
        return DEVICE
    
    def _step(self, model, data, seq, fgp):
        if self.config['fgp_flag']: 
            recon_logits, mask_pos, mu, log_var = model(data, seq[:, :-1], fgp)
        else:
            recon_logits, mask_pos, mu, log_var = model(data, seq[:, :-1])

        # print(f"Mean of latent vector: {mu.mean().item():.4f}")
        # print(f"Variance of latent vector: {log_var.exp().mean().item():.4f}")  # exp(log_var) 是标准差

        recon_loss, kl_loss, mmd_loss = mask_predict_vae_loss(recon_logits, pad_token_id=self.vocab.pad_idx, target=seq[:,1:], mu=mu, log_var=log_var, mask_pos=mask_pos, free_bits=0.1)

        return recon_loss, kl_loss, mmd_loss
    
    def _get_model(self):

        if self.config['pretrained_model'] == 'MolCLR':
            from models.gin import GINet
            graphmodel = GINet(num_layer=5, emb_dim=300, drop_ratio=0.3, pool='mean')
            checkpoints_folder = './models/ckpt/pretrained_gin/'
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device, weights_only=True)
            graphmodel.load_my_state_dict(state_dict)
        elif self.config['pretrained_model'] == 'MoleBERT':
            from models.model import GNN
            graphmodel = GNN(num_layer=5,emb_dim=300, drop_ratio=0.5, JK='last', gnn_type='gin')
            checkpoints = './models/ckpt/Mole-BERT.pth'
            state_dict = torch.load(checkpoints, map_location=self.device, weights_only=True)
            graphmodel.load_state_dict(state_dict)
        elif self.config['pretrained_model'] == 'MGSSL':
            from models.model import GNN
            graphmodel = GNN(num_layer=5, emb_dim=300, drop_ratio=0.5, JK='last', gnn_type='gin')
            checkpoints = './models/ckpt/MGSSL.pth'
            state_dict = torch.load(checkpoints, map_location=self.device, weights_only=True)
            graphmodel.load_state_dict(state_dict)
        elif self.config['pretrained_model'] == 'MRCD':
            state_dict_path = './models/ckpt/CLR_model_MolCLR.pth'
            CLRmodel = torch.load(state_dict_path, map_location=self.device, weights_only=False)
            graphmodel = CLRmodel.graphmodel
        else:
            raise ValueError(f"No pretrainedmodel: {self.config['pretrained_model']}")
       
        encoder = GraphEncoder(graphmodel, latent_dim = self.config['decoder']['latent_dim'],finetune=self.config['finetune_flag'])
        decoder = SMILESDecoder(vocab=self.vocab, max_length=self.config['max_length'], **self.config['decoder'])
        vae = VAE(encoder=encoder, decoder=decoder)
        return vae
    
    def _get_optimizer(self, config, model, train_loader):
        optimizer_groups = [
    {'params': model.encoder.pretrained_model.parameters(), 
    'lr': config['init_encoder_lr'], 
    'weight_decay':0},
    
    {"params": model.encoder.mu_layer.parameters(),
     "lr": config['init_decoder_lr'],
     "weight_decay": config['weight_decay']},
    
    {"params": model.encoder.log_var_layer.parameters(),
     "lr": config['init_decoder_lr'] ,
     "weight_decay": config['weight_decay']},
    
    {'params': model.decoder.parameters(), 
    'lr': config['init_decoder_lr'], 
    'weight_decay':config['weight_decay']},
]

        if config['optimizer'] == 'adam':
            optimizer = Adam(
                optimizer_groups,
                betas=(0.95, 0.999),  
                eps=1e-7
            )
        elif config['optimizer'] == 'adamw':
            optimizer = AdamW(
                optimizer_groups,
                betas=(0.95, 0.999),  
                eps=1e-7
            )
        elif config['optimizer'] == 'sgd':
            optimizer = SGD(
                optimizer_groups,
                momentum=0.9, 
                nesterov=True
            )
        
        steps_per_epoch = len(train_loader)
        training_steps = steps_per_epoch * config['epochs']
        warmup_steps = int(training_steps * config['warmup_ratio'])

       
        scheduler = get_scheduler(
                optimizer = optimizer, name = config['scheduler'], num_warmup_steps = warmup_steps, num_training_steps=training_steps
            )

        
        return optimizer, scheduler

    def _save_config_file(self, model_checkpoints_folder):
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
            shutil.copy('configs/vae_config.yaml', os.path.join(model_checkpoints_folder, 'vae_config.yaml'))
    
    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        model = self._get_model().to(self.device)
        optimizer, scheduler = self._get_optimizer(self.config, model, train_loader)
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        self._save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch in range(self.config['epochs']):
            tot_epoch_loss = 0
            tot_recon_loss = 0
            tot_kl_loss = 0
            tot_mmd_loss = 0
            batch_idx = 0
            for data, seq, fgp in tqdm(train_loader,desc=f'Training Epoch {epoch+1}'):
                optimizer.zero_grad()

                data = data.to(DEVICE)
                seq = seq.to(DEVICE)
                fgp = fgp.to(DEVICE)
                
                recon_loss, kl_loss, mmd_loss = self._step(model, data, seq, fgp)
                kl_weight = KLAnnealer(self.config).get_kl_weight(epoch)
                loss = recon_loss + kl_weight * kl_loss + 2 * (1-kl_weight) * mmd_loss

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('recon_loss', recon_loss, global_step=n_iter)
                    self.writer.add_scalar('kl_loss', kl_loss, global_step=n_iter)
                    self.writer.add_scalar('kl_weight', kl_weight, global_step=n_iter)
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                # if batch_idx ==0 and epoch==0:
                #     print("\n--- First Batch Debug ---")
                #     print(f"Pre-update KL: {kl_loss.item():.3e}")
                #     print("Gradients before clip:")
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             print(f"{name}: {param.grad.abs().mean():.3e}")
                    
                #     loss.backward()
                    
                    
                #     print("\nGradients after backward:")
                #     for name, param in model.named_parameters():
                #         if param.grad is not None:
                #             print(f"{name}: {param.grad.abs().mean():.3e}")
                #     torch.nn.utils.clip_grad_value_(model.parameters(), 1)
                #     optimizer.step()
                    
                #     # 立即检查参数更新量
                #     with torch.no_grad():
                #         delta = []
                #         for name, param in model.named_parameters():
                #             if param.grad is not None:
                #                 delta.append(param.data.abs().mean().item())
                #         print(f"Param delta mean: {np.mean(delta):.3e}")
                    
                #     exit() 
                
                loss.backward()
                clip_grad_norm_(model.parameters(),max_norm=self.config['grad_clip'])
                

                optimizer.step()
                scheduler.step()

                tot_epoch_loss += loss.item()
                tot_recon_loss += recon_loss.item()
                tot_kl_loss += kl_loss.item()
                tot_mmd_loss += mmd_loss.item()
                n_iter += 1
                batch_idx += 1

            print(
                f"Epoch [{epoch+1}/{self.config['epochs']}], Recon_loss:{tot_recon_loss/(batch_idx):.4f}, KL_loss:{tot_kl_loss/(batch_idx):.4f}, Beta:{kl_weight}, MMD_loss:{mmd_loss/(batch_idx):.4f}, Loss: {tot_epoch_loss/(batch_idx):.4f}"
            )

            if epoch % self.config['eval_every_n_epochs'] == 0:
                profile = self._generate(model)
                print(f"Epoch [{epoch+1}/{self.config['epochs']}], Validity: {profile['validity']:.4f}, Uniqueness: {profile['uniqueness']:.4f}")

            valid_recon_loss, valid_kl_loss, valid_loss = self._evaluate(model, valid_loader, kl_weight)
            # print(f"Epoch [{epoch+1}/{self.config['epochs']}], Valid_loss: {valid_loss:.4f}, Valid_recon_loss: {valid_recon_loss:.4f}, Valid_kl_loss: {valid_kl_loss:.4f}")
            self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
            valid_n_iter += 1

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'best_model.pth'))
                print(f'model saved with valid loss: {valid_loss} at {model_checkpoints_folder},KL_loss:{valid_kl_loss},Recon_loss:{valid_recon_loss}')
            
        self._test(model, test_loader, kl_weight)
    def _generate(self, model, num_samples=1000, num_iters=5):   
        model.eval()
        valid = 0
        unique = set()
        
        with torch.no_grad():
            z = torch.randn(num_samples, model.decoder.latent_dim).to(self.device)
            
            sampled_tokens = model.decoder._generate_forward(z, num_iters)

            # sampled_tokens = torch.multinomial(prob.view(-1, prob.size(-1)), 1).view(prob.shape[:-1])
            generated_smiles = [self.vocab.decode(seq.cpu().numpy()) for seq in sampled_tokens]
            
            
            for smi in generated_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None: 
                    valid += 1
                    unique.add(smi)
                    
        return {
            "validity": valid/num_samples,
            "uniqueness": len(unique)/num_samples,
            # "novelty": len([s for s in unique if s not in training_set])/len(unique)
        }
    def _evaluate(self, model, loader, kl_weight):
        model.eval()
        total_loss = 0
        batch_idx = 0

        with torch.no_grad():
            for data, seq, fgp in tqdm(loader, desc='Evaluating'):
                data = data.to(self.device)
                seq = seq.to(self.device)
                fgp = fgp.to(self.device)

                recon_loss, kl_loss, mmd_loss = self._step(model, data, seq, fgp)
                loss = recon_loss + kl_weight * kl_loss + 5 * (1-kl_weight) * mmd_loss

                total_loss += loss.item()
                batch_idx += 1
        return recon_loss, kl_loss, total_loss / batch_idx
    
    def _test(self, model, loader, kl_weight):
        model.eval()
        with torch.no_grad():
            for data, seq, fgp in tqdm(loader, desc='Testing'):
                data = data.to(self.device)
                seq = seq.to(self.device)
                fgp = fgp.to(self.device) 

                recon_loss, kl_loss, mmd_loss = self._step(model, data, seq, fgp)
                loss = recon_loss + kl_weight * kl_loss + 5 * (1-kl_weight) * mmd_loss

                total_loss += loss.item()
                batch_idx += 1
        print(f"Test loss: {total_loss / batch_idx:.4f}, Recon_loss: {recon_loss:.4f}, KL_loss: {kl_loss:.4f}")

            

def set_random(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed) # PyTorch multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   

def main(config):
    print('Loading vocab ...')
    vocab = load_vocab_from_pickle(config['vocab_path'])
    print('The size of vocab:', len(vocab))
    dataset = MoleculeLoaderWrapper(data_path = config['data_path'], vocab = vocab, batch_size = config['batch_size'], **config['dataset'])
    trainer = VAE_trainer(config = config, dataset=dataset, vocab=vocab)
    trainer.train()

if __name__ == "__main__":
    config = yaml.load(open("configs/vae_config.yaml", "r"), Loader=yaml.FullLoader)
    seed = config['random_seed']
    set_random(seed)
    print(config)
    main(config)
