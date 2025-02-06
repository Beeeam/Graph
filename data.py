import csv
import math
import time
import h5py
import random
from umap.umap_ import UMAP
import pandas as pd
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data.sampler import SubsetRandomSampler

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, Dataset


from utils import molecule_to_graph

"""Adapted from:https://github.com/yuyangw/MolCLR/blob/master/dataset/dataset_test.py"""

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data

class MoleculeDataset_h5(Dataset):
    def __init__(self, h5_path, vocab):
        super(MoleculeDataset_h5, self).__init__()
        self.h5 = h5py.File(h5_path, 'r')['data']
        self.vocab = vocab
        self.length = self.h5['seq'].shape[0]

    def __getitem__(self, idx):

        x = self.h5['x'][idx]          # shape: (num_nodes * 2,)
        edge_index = self.h5['edge_index'][idx]  # shape: (2 * num_edges,)
        edge_attr = self.h5['edge_attr'][idx]    # shape: (num_edges * 2,)

        num_nodes = len(x) // 2
        num_edges = len(edge_index) // 2

        data = Data(
            x=torch.tensor(x).view(num_nodes, 2).long(), 
            edge_index=torch.tensor(edge_index).view(2, num_edges).long(),
            edge_attr=torch.tensor(edge_attr).view(num_edges, 2).long()
        )

        seq_np = self.h5['seq'][idx]
        seq = torch.from_numpy(seq_np).long()

        fgp_np = self.h5['fgp'][idx]
        fgp = torch.from_numpy(fgp_np).long()

        return data, seq, fgp

    def __len__(self):
        return self.length

def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset.smiles_data):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds = []
    valid_inds = []
    test_inds = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds

class MoleculeLoaderWrapper(object):
    
    def __init__(self, 
        batch_size, h5_path, num_workers, valid_size, test_size, 
         vocab, splitting, max_seq_length = 128, fpSize = 2048, seed = None
    ):
        super(object, self).__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.vocab = vocab
        self.splitting = splitting
        self.max_seq_length = max_seq_length
        self.fpSize = fpSize
        self.seed = seed if seed is not None else 42
        np.random.seed(self.seed)
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        train_dataset = MoleculeDataset_h5(h5_path=self.h5_path, vocab=self.vocab)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            print('Dataset is random splitting...')
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)

            split = int(np.floor(self.valid_size * num_train))
            split2 = int(np.floor(self.test_size * num_train))
            valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]
        
        elif self.splitting == 'scaffold':
            print('Dataset is scaffold splitting...')
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        def collate_fn(batch):
            graphs, seqs ,fgps = zip(*batch)

            batch_graph = Batch.from_data_list(graphs)
            padded_seqs = torch.nn.utils.rnn.pad_sequence(
                seqs, 
                batch_first=True, 
                padding_value=self.vocab.pad_idx
            )

            fgps_tensor = torch.stack(fgps, dim=0)

            return batch_graph, padded_seqs, fgps_tensor

        common_args = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'collate_fn': collate_fn,
            'pin_memory': True,
            'drop_last': False
            
        }  


        train_loader = DataLoader(
            train_dataset, sampler=train_sampler,**common_args
        )
        valid_loader = DataLoader(
            train_dataset, sampler=valid_sampler, **common_args
        )
        test_loader = DataLoader(
            train_dataset, sampler=test_sampler, **common_args
        )

        return train_loader, valid_loader, test_loader
