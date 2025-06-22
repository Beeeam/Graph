import math
import time
import random
import pandas as pd
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics import R2Score, MeanSquaredError
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj, scatter)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from torch_geometric.data import Data, Dataset, DataLoader


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



class PolyDataset(Dataset):
    def __init__(self, data, targets):
        super(PolyDataset, self).__init__()
        self.data = data
        self.targets = [targets] if isinstance(targets, str) else targets
        # print(self.targets)
        assert all(target in ['CO2', 'O2', 'N2', 'CO2/O2', 'CO2/N2'] for target in self.targets)
        self.smiles_data = self.data['SMILES'].values
        

    def __getitem__(self, idx):
        molecule_str = self.data['SMILES'].values[idx]
        mol = Chem.MolFromSmiles(molecule_str)
        if mol is None:
            raise ValueError(f"wrong SMILES: {molecule_str}")
        # mol = Chem.AddHs(mol)
        generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprint = generator.GetFingerprint(mol)

        # umap = UMAP(n_components=32, random_state=42, n_neighbors=50)
        # fingerprint = umap.fit_transform(fingerprint)
        fingerprint_tensor = torch.tensor(list(fingerprint), dtype=torch.float32)


        chirality_idx = []
        atomic_number = []
        # aromatic = []
        # sp, sp2, sp3, sp3d = [], [], [], []
        # num_hs = []
        for atom in mol.GetAtoms():
            atomic_number.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            # aromatic.append(1 if atom.GetIsAromatic() else 0)
            # hybridization = atom.GetHybridization()
            # sp.append(1 if hybridization == HybridizationType.SP else 0)
            # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

        # z = torch.tensor(atomic_number, dtype=torch.long)
        x1 = torch.tensor(atomic_number, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)
        # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
        #                     dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)
        
        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
        # print(f"edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}")

        # kernel_param = range(1,21)
        # N = x.shape[0]
        # rw_landing, edge_features = get_rw_landing_probs_and_edge_features(ksteps=kernel_param,
        #                                   edge_index=edge_index,
        #                                   num_nodes=N)
        # # print(edge_features)
        # if edge_attr.dim() == 1:
        #     edge_attr = torch.cat([edge_attr.unsqueeze(dim=-1), edge_features], dim=1)
        # else:
        #     edge_attr = torch.cat([edge_attr, edge_features], dim=1)
        # print(f"edge_attr: {edge_attr.shape}")

        labels = self.data.iloc[idx][self.targets].values
        # labels = (labels - self.mean) / self.std
        labels_tensor = torch.tensor(labels.astype(np.float32), dtype=torch.float32)
        
        for i, target in enumerate(self.targets):
            if target in ['CO2', 'O2', 'N2']:
                labels_tensor[i] = torch.log10(labels_tensor[i])

        
        data = Data(x=x, y=labels_tensor, edge_index=edge_index, edge_attr=edge_attr)


        
        return data, fingerprint_tensor

    def __len__(self):
        return len(self.data)

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

class PolyGNNLoaderWrapper(object):
    
    def __init__(self, 
        batch_size, train_data_path, valid_data_path, test_data_path,
        num_workers, valid_size,  
        test_size, splitting, targets, seed = None
    ):
        super(object, self).__init__()
        self.train_data = pd.read_csv(train_data_path)
        self.valid_data = pd.read_csv(valid_data_path)
        self.test_data = pd.read_csv(test_data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.targets = targets
        # self.task = task
        self.splitting = splitting
        self.seed = seed if seed is not None else 42
        np.random.seed(self.seed)
        assert splitting in ['random', 'scaffold']

    def get_data_loaders(self):
        train_dataset = PolyDataset(data=self.train_data, targets=self.targets)
        valid_dataset = PolyDataset(data=self.valid_data, targets=self.targets)
        test_dataset = PolyDataset(data=self.test_data, targets=self.targets)
        train_loader, valid_loader, test_loader = self.get_train_validation_data_loaders(train_dataset, valid_dataset, test_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_validation_data_loaders(self, train_dataset, valid_dataset, test_dataset):
        if self.splitting == 'random':
            # obtain training indices that will be used for validation
            train_loader = DataLoader(
                            train_dataset, batch_size=self.batch_size, shuffle=True, 
                            num_workers=self.num_workers, drop_last=False
                        )
            valid_loader = DataLoader(
                            valid_dataset, batch_size=self.batch_size, shuffle=True, 
                            num_workers=self.num_workers, drop_last=False
                        )
            test_loader = DataLoader(
                            test_dataset, batch_size=self.batch_size, shuffle=True, 
                            num_workers=self.num_workers, drop_last=False
                        )
        
        elif self.splitting == 'scaffold':
            print('Dataset is scaffold splitting...')
            train_dataset = ConcatDataset([train_dataset, valid_dataset, test_dataset])
            train_idx, valid_idx, test_idx = scaffold_split(train_dataset, self.valid_size, self.test_size)

        # define samplers for obtaining training and validation batches
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                num_workers=self.num_workers, drop_last=False
            )
            valid_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                num_workers=self.num_workers, drop_last=False
            )
            test_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, sampler=test_sampler,
                num_workers=self.num_workers, drop_last=False
            )

        return train_loader, valid_loader, test_loader


def get_rw_landing_probs_and_edge_features(ksteps, edge_index, edge_weight=None,
                                           num_nodes=None, space_dim=0):
    """
    https://github.com/LUOyk1999/GNNPlus/blob/main/GNNPlus/transform/posenc_stats.py
    Compute Random Walk landing probabilities for given list of K steps,
    and extract random walk probabilities for edges defined in edge_index.
    
    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
    
    Returns:
        rw_landing: 2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
        edge_features: Tensor with shape (num_edges, len(ksteps)), RW probs for edges
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    # Transition matrix P = D^-1 * A
    adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes).squeeze(0)  # (Num nodes) x (Num nodes)
    P = torch.diag(deg_inv) @ adj  # Transition matrix

    rws = []
    edge_features = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            edge_features.append(Pk[source, dest] * (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            Pk = P.matrix_power(k)
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            edge_features.append(Pk[source, dest] * (k ** (space_dim / 2)))

    rw_landing = torch.stack(rws, dim=1)  # (Num nodes) x (K steps)
    edge_features = torch.stack(edge_features, dim=1)  # (Num edges) x (K steps)
    edge_features = edge_features.long()  # Convert to long type for consistency

    return rw_landing, edge_features