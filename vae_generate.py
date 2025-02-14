import os
import random
import yaml
import numpy as np

from rdkit import Chem

import torch
import torch.nn.functional as F


from vae import GraphEncoder, SMILESDecoder, VAE
from data import MoleculeLoaderWrapper
from utils import vae_loss, load_vocab_from_pickle


DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

class vaegenerator(object):
    def __init__(self, config, dataset, vocab):
        self.config = config
        self.dataset = dataset
        self.vocab = vocab

        self.device = self._get_device()
        self.metric = 0
        self.valid_generated = []

    def _get_device(self):
        return DEVICE

    def _step(self, model, data, seq, fgp):
        recon_logits, mu, log_var = model(data, seq[:, :-1])

        recon_loss, kl_loss = vae_loss(recon_logits, target=seq[:,1:], mu=mu, log_var=log_var)

        return recon_loss, kl_loss
    
    def _get_model(self):

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
        elif self.config['pretrained_model'] == 'MRCD':
            state_dict_path = './models/ckpt/CLR_model_MolCLR.pth'
            CLRmodel = torch.load(state_dict_path, map_location=DEVICE, weights_only=False)
            graphmodel = CLRmodel.graphmodel
        else:
            raise ValueError(f"No pretrainedmodel: {self.config['pretrained_model']}")
        
        encoder = GraphEncoder(graphmodel, latent_dim=self.config['decoder']['latent_dim'], finetune=self.config['finetune_flag'])
        decoder = SMILESDecoder(vocab=self.vocab, max_length=self.config['max_length'], **self.config['decoder'])
        vae = VAE(encoder=encoder, decoder=decoder)
        return vae
    
    def test(self):
        model = self._get_model()
        model.load_state_dict(torch.load(self.config['model_path'], map_location=self.device)).to(self.device)
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        if self.config['mode'] == 'test':
            self.mertic = self._evaluate(model, test_loader)
            print(f'Reconstruction Accuracy:{self.metric}')

        elif self.config['mode'] == 'generate':
            self.metric = self._generate(model, num_samples=self.config['num_samples'],temperature=self.config['temperature'])
            print(f'Valid Rate:{self.metric}')
    
    def _evaluate(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, seq, fgp) in enumerate(data_loader):
                data, seq = data.to(self.device), seq.to(self.device)
                
                recon_logits, mu, log_var = model(data, seq[:, :-1])

                recon_token = torch.argmax(recon_logits, dim=-1)
                recon_smiles = [self.vocab.decode(s) for s in recon_token.cpu().numpy()]
                target_smiles = [self.vocab.decode(s) for s in seq.cpu().numpy()]
                for pred, target in zip(recon_smiles, target_smiles):
                    if pred == target:
                        self.valid_generated.append(pred)
                        correct += 1
                    total += 1
        print(f'Successfully reconstructed {correct}/{total} SMILES')
        return correct / total

    def _generate(self, model, num_samples, temperature):
        model.eval()
        valid = 0
        valid_rate = 0
        with torch.no_grad():
            z = torch.randn(num_samples, model.decoder.latent_dim).to(self.device)
            
            logits = model.decoder(z)
            prob = F.softmax(logits/temperature, dim=-1)

            top_k = self.config['top_k']
            top_k_probs, top_k_indices = torch.topk(prob, top_k, dim=-1)
            samples = torch.multinomial(top_k_probs.view(-1, top_k), 1).view(prob.shape[:-1])
            sampled_tokens = torch.gather(top_k_indices, -1, samples.unsqueeze(-1)).squeeze(-1)

            # sampled_tokens = torch.multinomial(prob.view(-1, prob.size(-1)), 1).view(prob.shape[:-1])
            generated_smiles = [self.vocab.decode(seq.cpu().numpy()) for seq in sampled_tokens]
            
            
            for smi in generated_smiles:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    valid += 1
                    self.valid_generated.append(Chem.MolToSmiles(mol))
            valid_rate = valid/num_samples
            return valid_rate

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
    generator = vaegenerator(config, dataset, vocab)
    generator.test()
    return generator.valid_generated 

if __name__ == '__main__':
    config = yaml.load(open("configs/vae_test_config.yaml", "r"), Loader=yaml.FullLoader)
    seed = config['random_seed']
    set_random(seed)
    print(config)
    smiles = main(config)
    print(smiles)