from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
import torch.nn.init as init
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

class CLRmodel(nn.Module):

    def __init__(self, seqmodel, graphmodel):
        super(CLRmodel, self).__init__()
        self.seqmodel = deepcopy(seqmodel)
        self.graphmodel = deepcopy(graphmodel)
        for param in self.seqmodel.parameters():
            param.requires_grad = False

        self.seq_dim = self.seqmodel.config.hidden_size
        self.grp_dim = self.graphmodel.emb_dim

        self.projector_fgp = nn.Linear(2048, self.grp_dim)
        self.projector_seq = nn.Linear(self.seq_dim, self.grp_dim)


        self.mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.grp_dim * 2, self.grp_dim),
            nn.ReLU(),
            nn.Linear(self.grp_dim, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, graph, fingerprint, encoding):
        xis = self.graphmodel(graph)

        sequence = self.seqmodel(input_ids=encoding['input_ids'].to(DEVICE), attention_mask=encoding['attention_mask'].to(DEVICE)).last_hidden_state
        sequence = sequence[:, 0, :].squeeze(0).clone().detach()
        fgp = self.projector_fgp(fingerprint)
        seq = self.projector_seq(sequence)
        # print(fgp.shape)
        # print(seq.shape)

        score = 1

        xjs = score * seq + (1 - score) * fgp
        # print(xis.shape)

        
        # print(xjs.shape)

        return xis, xjs