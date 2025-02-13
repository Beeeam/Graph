import re
import csv
import h5py
import math
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Set, Dict

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT

import torch
import torch.nn as nn
from torch_geometric.data import Data

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

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

def molecule_to_graph(mol):
    chirality_idx = []
    atomic_number = []

    for atom in mol.GetAtoms():
        atomic_number.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x1 = torch.tensor(atomic_number, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x = torch.cat([x1, x2], dim=-1)


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
    edge_attr = torch.tensor(edge_feat, dtype=torch.long)
  
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def calculate_ECFP(mol, radius=2, fpSize=2048):
    generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    fingerprint = generator.GetFingerprint(mol)
    return list(fingerprint)

def vae_loss(recon_output, target, mu, log_var):
    criterion = nn.CrossEntropyLoss()
    recon_loss = criterion(recon_output.reshape(-1, recon_output.size(-1)), target.reshape(-1))
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss, kl_loss

class KLAnnealer:
    def __init__(self, config):
        self.kl_start = config['kl_anneal']['kl_start']
        self.kl_end = config['kl_anneal']['kl_end']
        self.kl_anneal_epochs = config['kl_anneal']['kl_anneal_epochs']
        self.mode = config['kl_anneal']['mode']
        self.cycles = 5

    def get_kl_weight(self, epoch):
        if epoch >= self.kl_anneal_epochs:
            return self.kl_end
        
        if self.mode in ['linear', 'sigmoid', 'exp', 'cosine']:
            y = epoch / self.kl_anneal_epochs
        elif self.mode == 'cyclical':
            # Cyclical annealing
            cycle_length = max(self.kl_anneal_epochs // self.cycles,1)
            epoch_in_cycle = epoch % cycle_length
            y = epoch_in_cycle / cycle_length
        else:
            raise ValueError("Invalid mode! Choose from 'linear', 'sigmoid', 'exp', 'cosine', or 'cyclical'.")

        if self.mode == 'linear':
            return self.kl_start + (self.kl_end - self.kl_start) * y
        elif self.mode == "sigmoid":
            return self.kl_end / (1 + np.exp(-10 * (y - 0.5)))  # Sigmoid function
        elif self.mode == "exp":
            if self.kl_start == 0:
                return self.kl_end * y
            return self.kl_start * ((self.kl_end / self.kl_start) ** y)  # Exponential
        elif self.mode == 'cosine':
            slop = 0.5 * (1 - math.cos(math.pi * y))
            return self.kl_start + (self.kl_end - self.kl_start) * slop
        elif self.mode == 'cyclical':
            return self.kl_start + (self.kl_end - self.kl_start) * y

    
class SpecialTokens:
    bos = '<bos>'
    eos = '<eos>'
    pad = '<pad>'
    unk = '<unk>'

    @classmethod
    def all(cls) -> List[str]:
        return [cls.bos, cls.eos, cls.pad, cls.unk]

class SMILESRegex:
    pattern = re.compile(
        r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|S|B|C|N|O|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|H|\[|\])"
    )

class SMILESVocab:
    def __init__(self, tokens: Set[str], special_tokens: SpecialTokens = SpecialTokens):
        
        for st in special_tokens.all():
            if st in tokens:
                raise ValueError(f"Special token {st} found in tokens")

        all_tokens = sorted(tokens) + special_tokens.all()
        self._st = special_tokens
        self.token2idx: Dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.idx2token: Dict[int, str] = {i: t for i, t in enumerate(all_tokens)}

       
        self.tokenizer = SMILESRegex.pattern.findall

    @classmethod
    def from_smiles_list(cls, smiles_list: List[str], special_tokens: SpecialTokens = SpecialTokens):
        tokens = set()
        for smi in smiles_list:
            tokens.update(cls.tokenize_smiles(smi))
        return cls(tokens, special_tokens)

    @staticmethod
    def tokenize_smiles(smiles: str) -> List[str]:

        return SMILESRegex.pattern.findall(smiles)

    @property
    def bos_token(self) -> str:
        return self._st.bos

    @property
    def eos_token(self) -> str:
        return self._st.eos

    @property
    def pad_token(self) -> str:
        return self._st.pad

    @property
    def unk_token(self) -> str:
        return self._st.unk

    @property
    def bos_idx(self) -> int:
        return self.token2idx[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.eos_token]

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.pad_token]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.unk_token]

    def __len__(self) -> int:
        return len(self.token2idx)

    def token_to_idx(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_idx)

    def idx_to_token(self, idx: int) -> str:
        return self.idx2token.get(idx, self.unk_token)

    def encode(
        self, 
        smiles: str, 
        add_bos: bool = False, 
        add_eos: bool = False,
        max_length: int = None,
        padding: bool = False
    ) -> List[int]:
        tokens = self.tokenize_smiles(smiles)
        indices = [self.token_to_idx(t) for t in tokens]
        
        if add_bos:
            indices = [self.bos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]
        
        if max_length is not None:
            if padding and len(indices) < max_length:
                indices += [self.pad_idx] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
        
        return indices

    def decode(
        self, 
        indices: List[int], 
        rem_bos: bool = True, 
        rem_eos: bool = True,
        rem_pad: bool = True
    ) -> str:
        tokens = []
        for idx in indices:
            token = self.idx_to_token(idx)
            if rem_pad and token == self.pad_token:
                continue
            tokens.append(token)
        
        if rem_bos and tokens and tokens[0] == self.bos_token:
            tokens = tokens[1:]
        if rem_eos and tokens and tokens[-1] == self.eos_token:
            tokens = tokens[:-1]
        
        return "".join(tokens)

    def batch_encode(
        self,
        smiles_list: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_length: int = None,
        padding: bool = False,
        return_tensor: bool = False
    ) -> List[List[int]]:
        encoded = [
            self.encode(smi, add_bos, add_eos, max_length, padding) 
            for smi in smiles_list
        ]
        if return_tensor:
            return torch.tensor(encoded, dtype=torch.long)
        return encoded

    def batch_decode(
        self, 
        indices_list: List[List[int]], 
        **kwargs
    ) -> List[str]:
        return [self.decode(indices, **kwargs) for indices in indices_list]

    def to_onehot(self, indices: List[int]) -> torch.Tensor:
        vec = torch.zeros(len(indices), len(self), dtype=torch.float)
        for i, idx in enumerate(indices):
            vec[i, idx] = 1.0
        return vec

    def get_properties(self) -> Dict:
        return {
            "vocab_size": len(self),
            "special_tokens": self._st.all(),
            "chemical_tokens": [t for t in self.token2idx.keys() if t not in self._st.all()]
        }
    
    
    def print_summary(self, max_tokens=20):
        print(f"Vocab Size: {len(self)}")
        print("Special Tokens:")
        for st in self._st.all():
            print(f"  {st}: {self.token2idx[st]}")
        
        chem_tokens = [t for t in self.token2idx if t not in self._st.all()]
        print(f"\nFirst {max_tokens} Chemical Tokens:")
        for t in chem_tokens[:max_tokens]:
            print(f"  {t}: {self.token2idx[t]}")

    def save_vocab(self, filepath: str):

        with open(filepath, 'w') as f:
            for t, idx in sorted(self.token2idx.items(), key=lambda x: x[1]):
                f.write(f"{t}\t{idx}\n")

    @classmethod
    def load_vocab(cls, filepath: str, special_tokens=SpecialTokens):

        tokens = []
        with open(filepath, 'r') as f:
            for line in f:
                t, _ = line.strip().split('\t')
                tokens.append(t)
        # 过滤特殊符号
        chem_tokens = [t for t in tokens if t not in special_tokens.all()]
        return cls(set(chem_tokens), special_tokens)

def build_vocab_large(data_path, chunk_size=1e6, save_path='pubchem_vocab.pkl'):
    token_counter = defaultdict(int)
    
    with open(data_path, 'r') as f:
        for line in tqdm(f, desc="Building Vocab"):
            smi = line.strip()
            # print(smi)
            tokens = SMILESVocab.tokenize_smiles(smi)
            for t in tokens:
                token_counter[t] += 1
                
            
            if len(token_counter) > chunk_size:
                
                freqs = sorted(token_counter.items(), key=lambda x: -x[1])
                cumsum = 0
                total = sum(token_counter.values())
                for i, (t, cnt) in enumerate(freqs):
                    cumsum += cnt
                    if cumsum > 0.95 * total:
                        break
                token_counter = dict(freqs[:i])
    
    vocab = SMILESVocab(set(token_counter.keys()))
    with open(save_path, "wb") as f:
        pickle.dump(vocab, f)
    return vocab

def load_vocab_from_pickle(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return vocab

def preprocess_to_h5(data_path, vocab, h5_path, max_seq_len=128, fpSize=2048):
    with h5py.File(h5_path, 'w') as f:
        
        grp = f.create_group('data')

        vlen_float = h5py.vlen_dtype(np.dtype('float32'))
        grp.create_dataset('x', shape=(0,), maxshape=(None,), dtype=vlen_float)
        grp.create_dataset('edge_index', shape=(0,), maxshape=(None,), dtype=vlen_float)
        grp.create_dataset('edge_attr', shape=(0,), maxshape=(None,), dtype=vlen_float)
        grp.create_dataset('seq', shape=(0, max_seq_len), maxshape=(None, max_seq_len), dtype=np.float32)
        grp.create_dataset('fgp', shape=(0, fpSize), maxshape=(None, fpSize), dtype=int)
        
        skipped = 0
        with open(data_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            idx = 0
            for row in tqdm(csv_reader, desc="Preprocessing"):
                smi = row[-1]
                try:
                    
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        raise ValueError(f"wrong SMILES: {smi}")
                    data = molecule_to_graph(mol)  
                    # print(f"Processing {idx}: SMILES: {smi}, Data.x shape: {data.x.shape}")
                    
                    
                    seq = vocab.encode(smi, max_length=max_seq_len, padding=True)
                    seq_tensor = torch.tensor(seq, dtype=torch.long)

                    fgp = calculate_ECFP(mol,fpSize=fpSize)
                    fgp_tensor = torch.tensor(fgp, dtype=torch.long)

                    if data.x.shape[0] == 0:  # If no node features
                        print(f"Skipping molecule {idx} (empty data.x)")
                        skipped += 1
                        continue
                    
                   
                    grp['edge_index'].resize((idx+1, ))
                    grp['edge_index'][idx] = data.edge_index.numpy().flatten().astype(np.float32)
                    
                    grp['edge_attr'].resize((idx+1, ))
                    grp['edge_attr'][idx] = data.edge_attr.numpy().flatten().astype(np.float32)
                    
                    grp['x'].resize((idx+1, ))
                    grp['x'][idx] = data.x.numpy().flatten().astype(np.float32)
                    
                    grp['seq'].resize((idx+1, max_seq_len))
                    grp['seq'][idx] = seq_tensor.numpy().astype(np.float32)

                    grp['fgp'].resize((idx+1, fpSize))
                    grp['fgp'][idx] = fgp_tensor.numpy()
                    
                    idx += 1
                except Exception as e:
                    print(f"Skipping due to error: {e}")
                    skipped += 1
        
        print(f"Total skipped molecules: {skipped}")

def MultiTaskMetrics(preds,labels):
    targets = ['CO2', 'O2', 'N2']
    metrics={
        'rmse': [],
        'r2': [],
        'pearson': []
    }
    task_metrics = dict()

    for i , target in enumerate(targets):
        y_pred = preds[:, i].detach().cpu().numpy()
        y_true = labels[:, i].detach().cpu().numpy()

        rmse = mean_squared_error(y_true, y_pred, squared=False)

        r2 = r2_score(y_true, y_pred)

        pearson_corr, _ = pearsonr(y_pred, y_true)

        metrics["rmse"].append(rmse)
        metrics["r2"].append(r2)
        metrics["pearson"].append(pearson_corr)

        task_metrics[f'task_{target}'] = {
            'rmse': rmse,
            'r2': r2,
            'pearson': pearson_corr
        }
    return task_metrics