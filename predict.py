import os
import shutil
import sys
import random
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
from torch.optim import AdamW, Adam, SGD
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import get_scheduler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from graph_loader import PolyGNNLoaderWrapper, PolyDataset
from predictor import Property_predictor
from utils import load_vocab_from_pickle

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('configs/predict_config.yaml', os.path.join(model_checkpoints_folder, 'finetune_config.yaml'))

class PredictorTrainer(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # dir_name = current_time + '_'  + '_' + config['dataset']['targets']
        self.task_names = config['dataset']['targets']
        if isinstance(self.task_names, list):
            targets_str = '-'.join(self.task_names)
        else:
            targets_str = self.task_names
            self.task_names = [self.task_names]

        dir_name = current_time + '__' + targets_str
        log_dir = os.path.join('ckpt/finetune', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset

        self.criterion = nn.MSELoss()

    def _get_device(self):
        return DEVICE
    
    def _step(self, model, data):
        preds = model(data) # CO2, O2, N2
        labels = data.y.view(preds.shape).to(torch.float32)
        loss = self.criterion(preds, labels)

        return loss
    
    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
               
        if self.config['pretrained_model'] == 'MolCLR':
            from models.gin import GINet
            graphmodel = GINet(num_layer=5, emb_dim=300, drop_ratio=0.3, pool='mean')
            checkpoints_folder = './models/ckpt/pretrained_gin/'
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=DEVICE, weights_only=True)
            graphmodel.load_my_state_dict(state_dict)
        elif self.config['pretrained_model'] == 'MoleBERT':
            from models.model import GNN
            graphmodel = GNN(num_layer=5,emb_dim=300, drop_ratio=0.5, JK='last', gnn_type='gin')
            checkpoints = './models/ckpt/Mole-BERT.pth'
            state_dict = torch.load(checkpoints, map_location=DEVICE, weights_only=True)
            graphmodel.load_state_dict(state_dict)
        elif self.config['pretrained_model'] == 'MGSSL':
            from models.model import GNN
            graphmodel = GNN(num_layer=5, emb_dim=300, drop_ratio=0.5, JK='last', gnn_type='gin')
            checkpoints = './models/ckpt/MGSSL.pth'
            state_dict = torch.load(checkpoints, map_location=DEVICE, weights_only=True)
            graphmodel.load_state_dict(state_dict)
        elif self.config['pretrained_model'] == 'CLR':
            from models.gin import GINet
            graphmodel = GINet(num_layer=5, emb_dim=300, drop_ratio=0.3, pool='mean') 
            state_dict_path = './models/ckpt/CLR_model_MolCLR.pth'
            # checkpoints = './models/ckpt/CLR_GIN.pth'
            CLRmodel = torch.load(state_dict_path, map_location=DEVICE, weights_only=False)
            # state_dict = torch.load(checkpoints, map_location=DEVICE, weights_only=True)
            # graphmodel.load_my_state_dict(state_dict)
            graphmodel = CLRmodel.graphmodel
        elif self.config['pretrained_model'] == 'scratch':
            from models.gin import GINet
            graphmodel = GINet(num_layer=5, emb_dim=300, drop_ratio=0.3, pool='mean')
        else:
            raise ValueError(f"No pretrainedmodel: {self.config['pretrained_model']}")
        
        predictor = Property_predictor(graphmodel, drop_out=self.config['drop_out'], finetune=self.config['finetune_flag'], num_tasks = len(self.task_names)).to(self.device)
        print('num_tasks:', len(self.task_names))
        if self.config['finetune_from'] is not None:
            self._load_pre_trained_weights(predictor)
        predictions = []
        labels = []

        predictor.eval()
        with torch.no_grad():
            test_loss = 0
            num_data = 0

            for data, fingerprint in tqdm(test_loader):
                data = data.to(self.device)
                pred = predictor(data)
                label = data.y.view(pred.shape).to(torch.float32)
                loss = self._step(predictor, data)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.device == 'cpu':
                    predictions.append(pred.detach().numpy())
                    # labels.extend(data.y.flatten().numpy())
                    # labels.append(data.y.cpu().numpy()) 
                    labels.append(label.cpu().numpy())
                else:
                    predictions.append(pred.cpu().detach().numpy())
                    # labels.extend(data.y.cpu().flatten().numpy())
                    # labels.append(data.y.cpu().numpy())
                    labels.append(label.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)  # shape: (total_samples, num_tasks)
        labels = np.concatenate(labels, axis=0)            # shape: (total_samples, num_tasks)

        if predictions.ndim == 1 or labels.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            labels = labels.reshape(-1, 1)
        # print('predictions shape:', predictions.shape, 'labels shape:', labels.shape)
        assert predictions.shape == labels.shape
        assert predictions.shape[1] == len(self.task_names)
        self.rmse = {name: mean_squared_error(labels[:, i], predictions[:, i]) ** 0.5
                    for i, name in enumerate(self.task_names)}
        self.r2 = {name: r2_score(labels[:, i], predictions[:, i])
                for i, name in enumerate(self.task_names)} 

        # print('Test loss:', test_loss / num_data, 'Test RMSE:', self.rmse, 'Test R2:', self.r2)
        print("Test RMSE:")
        for task, rmse in self.rmse.items():
            print(f"  {task}: {rmse:.4f}")
        print("Test R²:")
        for task, r2 in self.r2.items():
            print(f"  {task}: {r2:.4f}")
        # if self.config['mode'] == 'test':
        #     predictor.eval()
        #     with torch.no_grad():
        #         test_loss = 0
        #         num_data = 0

        #         for data, fingerprint in tqdm(train_loader):
        #             data = data.to(self.device)
        #             pred = predictor(data)
        #             label = data.y.view(pred.shape).to(torch.float32)
        #             loss = self._step(predictor, data)

        #             test_loss += loss.item() * data.y.size(0)
        #             num_data += data.y.size(0)

        #             if self.device == 'cpu':
        #                 predictions.extend(pred.detach().numpy())
        #                 labels.extend(data.y.flatten().numpy())
        #             else:
        #                 predictions.extend(pred.cpu().detach().numpy())
        #                 labels.extend(data.y.cpu().flatten().numpy())
            
        #     predictions = np.array(predictions)
        #     labels = np.array(labels)
            


        if self.config['optimizer'] == 'adam':
            optimizer = Adam(
                        [
                            {"params": predictor.Pretrained_model.parameters(), "lr": config['init_base_lr'],
                            "weight_decay": self.config['weight_decay']},
                            {"params": predictor.hidden_layer.parameters(), "lr": config['init_lr'],
                            "weight_decay": self.config['weight_decay']},
                            {"params": predictor.reg_layer.parameters(), "lr": config['init_lr'],
                            "weight_decay": self.config['weight_decay']},
                        ]
                    )
        elif self.config['optimizer'] == 'adamw':
            optimizer = AdamW(
                        [
                            {"params": predictor.Pretrained_model.parameters(), "lr": config['init_base_lr'],
                            "weight_decay": self.config['weight_decay']},
                            {"params": predictor.hidden_layer.parameters(), "lr": config['init_lr'],
                            "weight_decay": self.config['weight_decay']},
                            {"params": predictor.reg_layer.parameters(), "lr": config['init_lr'],
                            "weight_decay": self.config['weight_decay']},
                        ]
                    )
        
        steps_per_epoch = len(train_loader)
        training_steps = steps_per_epoch * self.config['epochs']
        warmup_steps = int(training_steps * self.config['warmup_ratio'])

        if self.config['scheduler'] == 'cosine':
            scheduler = get_scheduler(
                optimizer = optimizer, name = self.config['scheduler'], num_warmup_steps = warmup_steps, num_training_steps=training_steps
            )
        elif self.config['scheduler'] == 'linear':
            scheduler = get_scheduler(
                optimizer = optimizer, name = self.config['scheduler'], num_warmup_steps = warmup_steps, num_training_steps=training_steps
            )
        
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        patience = 10
        counter = 0
        early_stop = False
        best_valid_loss = np.inf
        best_valid_rmse = np.inf
        best_valid_r2 = -np.inf

        for epoch_counter in range(self.config['epochs']):
            tot_epoch_loss = 0
            batch_idx = 0
            predictor.train()
            for data, fingerprint in tqdm(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(predictor, data)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    # print(epoch_counter, loss.item())

                loss.backward()
               
                optimizer.step()
                scheduler.step()

                tot_epoch_loss += loss.item()
                n_iter += 1
                batch_idx += 1
            
            print(
                f"Epoch [{epoch_counter+1}/{config['epochs']}], Loss: {tot_epoch_loss/(batch_idx):.4f}"
            )

            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss, valid_rmse, valid_r2 = self._evaluate(predictor, valid_loader)
                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                for task_name in valid_rmse:
                    self.writer.add_scalar(f'valid_rmse/{task_name}', valid_rmse[task_name], global_step=valid_n_iter)
                    self.writer.add_scalar(f'valid_r2/{task_name}', valid_r2[task_name], global_step=valid_n_iter)

                valid_n_iter += 1
                # print(f'epoch: {epoch_counter}, valid loss: {valid_loss}, valid rmse: {valid_rmse}, valid R2: {valid_r2}')
            avg_valid_rmse = np.mean(list(valid_rmse.values()))
            avg_valid_r2 = np.mean(list(valid_r2.values()))
            if valid_loss < best_valid_loss - 1e-4:
                best_valid_loss = valid_loss
                best_valid_rmse = valid_rmse
                best_valid_r2 = valid_r2
                torch.save(predictor.state_dict(), os.path.join(model_checkpoints_folder, 'best_model.pth'))
                print(f'model saved with valid loss: {valid_loss:.4f}, avg RMSE: {avg_valid_rmse:.4f}, avg R2: {avg_valid_r2:.4f}')
                counter = 0
            else:
                counter += 1
                print(f"[Epoch {epoch_counter}] no improvement in valid loss for {counter} epoch(s).")
                if counter >= patience and epoch_counter >= 80:
                    print(f"Early stopping at epoch {epoch_counter}. Best valid loss: {best_valid_loss:.4f}")
                    early_stop = True
                    break
                
        self._test(predictor, test_loader)

    def _evaluate(self, model, valid_loader):
        model.eval()
        total_loss = 0
        rmse = 0
        total_num = 0
        predictions = []
        labels = []

        with torch.no_grad():
            for data, fingerprint in tqdm(valid_loader):

                data = data.to(self.device)
                pred = model(data)
                label = data.y.view(pred.shape).to(torch.float32)
                loss = self._step(model, data)
                total_loss += loss.item() * data.y.size(0)
                total_num += data.y.size(0)

                if self.device == 'cpu':
                    predictions.append(pred.detach().numpy())
                    # labels.extend(data.y.flatten().numpy())
                    # labels.append(data.y.cpu().numpy())
                    labels.append(label.cpu().numpy()) 
                else:
                    predictions.append(pred.cpu().detach().numpy())
                    # labels.extend(data.y.cpu().flatten().numpy())
                    # labels.append(data.y.cpu().numpy())
                    labels.append(label.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)  # shape: (total_samples, num_tasks)
        labels = np.concatenate(labels, axis=0)            # shape: (total_samples, num_tasks)

        if predictions.ndim == 1 or labels.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            labels = labels.reshape(-1, 1)
        assert predictions.shape == labels.shape
        assert predictions.shape[1] == len(self.task_names)
        rmse = {name: mean_squared_error(labels[:, i], predictions[:, i]) ** 0.5
                    for i, name in enumerate(self.task_names)}
        r2 = {name: r2_score(labels[:, i], predictions[:, i])
                for i, name in enumerate(self.task_names)} 


        return total_loss / total_num, rmse, r2
    
    def _test(self, model, test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'best_model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print('model loaded from:', model_path)

        predictions = []
        labels = []

        model.eval()
        with torch.no_grad():
            test_loss = 0
            num_data = 0

            for data, fingerprint in tqdm(test_loader):
                data = data.to(self.device)
                pred = model(data)
                label = data.y.view(pred.shape).to(torch.float32)
                loss = self._step(model, data)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.device == 'cpu':
                    predictions.append(pred.detach().numpy())
                    # labels.extend(data.y.flatten().numpy())
                    # labels.append(data.y.cpu().numpy()) 
                    labels.append(label.cpu().numpy())
                else:
                    predictions.append(pred.cpu().detach().numpy())
                    # labels.extend(data.y.cpu().flatten().numpy())
                    # labels.append(data.y.cpu().numpy())
                    labels.append(label.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)  # shape: (total_samples, num_tasks)
        labels = np.concatenate(labels, axis=0)            # shape: (total_samples, num_tasks)

        if predictions.ndim == 1 or labels.ndim == 1:
            predictions = predictions.reshape(-1, 1)
            labels = labels.reshape(-1, 1)
        assert predictions.shape == labels.shape
        assert predictions.shape[1] == len(self.task_names)
        self.rmse = {name: mean_squared_error(labels[:, i], predictions[:, i]) ** 0.5
                    for i, name in enumerate(self.task_names)}
        self.r2 = {name: r2_score(labels[:, i], predictions[:, i])
                for i, name in enumerate(self.task_names)} 

        print('Test loss:', test_loss / num_data, 'Test RMSE:', self.rmse, 'Test R2:', self.r2)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join(self.config['finetune_from'], 'checkpoints')
            stat_dict = torch.load(os.path.join(checkpoints_folder, 'best_model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            # pretrained_model = CLRmodel.graphmodel.state_dict()
            model.load_state_dict(stat_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

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
    dataset = PolyGNNLoaderWrapper(batch_size = config['batch_size'], **config['dataset'])
    # train_data = PolyDataset(config['train_data'], config['targets'])

    trainer = PredictorTrainer(dataset, config)
    trainer.train()

    return trainer.rmse, trainer.r2

if __name__ == "__main__":

    config = yaml.load(open("configs/predict_config.yaml", "r"), Loader=yaml.FullLoader)
    seed = config['random_seed']

    set_random(seed)

    target_list = ['CO2', 'O2', 'N2']

    print(config)

    results_list = []
    config['dataset']['targets'] = target_list
    rmse_dict, r2_dict = main(config)  # unpack

    for target in target_list:
        results_list.append([
            config['pretrained_model'],
            target,
            rmse_dict[target],
            r2_dict[target]
        ])
    # for target in target_list:
    #     config['dataset']['targets'] = target
    #     result = main(config)
    #     results_list.append([config['pretrained_model'], target, result])
    
    os.makedirs('experiments/predict', exist_ok = True)
    df = pd.DataFrame(results_list)
    df.to_csv(
        'experiments/predict/{}_predict.csv'.format(config['pretrained_model']), 
        mode='a', index=False, header=False
    )
    

