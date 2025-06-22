
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.nn.init as init

from copy import deepcopy

class Property_predictor(nn.Module):
    """multitask prediction models."""

    def __init__(self, pretrained_model, drop_out=0.3, finetune=True, num_tasks=1):
        super(Property_predictor, self).__init__()
        self.Pretrained_model = deepcopy(pretrained_model)
        if not finetune:
            for param in self.Pretrained_model.parameters():
                param.requires_grad = False
        hidden_size = self.Pretrained_model.emb_dim
        self.hidden_layer = nn.Sequential( 
            nn.LayerNorm(hidden_size),    
            # nn.Dropout(drop_out),      
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_out),
        )

        self.reg_layer = nn.Linear(hidden_size, num_tasks)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.hidden_layer:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
                init.zeros_(layer.bias)
        if isinstance(self.reg_layer, nn.Linear):
            init.kaiming_normal_(self.reg_layer.weight, mode='fan_in', nonlinearity='linear')
            init.zeros_(self.reg_layer.bias)



    def forward(self, data):
        x = self.Pretrained_model(data)
        res = x
        x = self.hidden_layer(x)
        x = x + res  # residual connection
        reg_out = self.reg_layer(x)

        return reg_out

class SAscore_predictor(nn.Module):
    """predictor for SA score"""

    def __init__(self, pretrained_model, drop_out, finetune=True):
        super(SAscore_predictor, self).__init__()
        self.Pretrained_model = deepcopy(pretrained_model)
        if not finetune:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        hidden_size = self.Pretrained_model.emb_dim
        self.hidden_layer = nn.Sequential(           
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(drop_out),
        )

        self.output_layer = nn.Linear(hidden_size//2, 1)

    def forward(self, data):
        x = self.Pretrained_model(data)
        x = self.hidden_layer(x)
        output = self.output_layer(x)
        return output