import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from copy import deepcopy
from vae_utils import top_k_sampling, top_p_sampling, beam_search

class GraphEncoder(nn.Module):
    def __init__(self,  pretrained_model, latent_dim = 128, finetune = False):
        super(GraphEncoder, self).__init__()
        self.pretrained_model = deepcopy(pretrained_model)
        if not finetune:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
        hidden_size = self.pretrained_model.emb_dim
        # self.mu_layer = nn.Linear(hidden_size, hidden_size)
        # self.log_var_layer = nn.Linear(hidden_size, hidden_size)
        self.mu_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, latent_dim))
        
        self.log_var_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_dim))
        
        nn.init.constant_(self.log_var_layer[-1].bias, -1) 
        nn.init.normal_(self.log_var_layer[-1].weight, mean=0, std=0.1)
        
        # for layer in [self.mu_layer, self.log_var_layer]:
        #     for m in layer.modules():
        #         if isinstance(m, nn.Linear):
        #             nn.init.normal_(m.weight, mean=0, std=0.01)
        #             if layer == self.mu_layer:
        #                 nn.init.constant_(m.bias, 0)
        #             else:
        #                 nn.init.constant_(m.bias, 0)
                    

    def forward(self, data):

        x = self.pretrained_model(data)

        mu = self.mu_layer(x)
        # mu = torch.tanh(mu) * 2
        log_var = self.log_var_layer(x)
        log_var = torch.clamp(log_var, min=-4, max=2)
        return mu, log_var
    
class SMILESDecoder(nn.Module):
    def __init__(self, vocab, max_length=128, latent_dim=128, embed_dim=128, dim_feedforward = 512, nhead=4, 
                 num_layers=2, mask_ratio=0.5):
        super(SMILESDecoder, self).__init__()
        
        self.vocab = vocab
        self.max_length = max_length
        self.vocab_size = len(vocab)
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        self.mask_token_id = vocab.mask_idx
        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx = vocab.pad_idx)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.3
            ),
            num_layers=num_layers
        )

        self.fc = nn.Linear(embed_dim, self.vocab_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(latent_dim, embed_dim)
        self.fgp_proj = nn.Linear(2048, embed_dim)
    
    def _generate_mask(self, bsz, seq_len, device):
        return (torch.rand(bsz, seq_len, device=device) < self.mask_ratio)
        
    def forward(self, z, tgt_seq = None, fgp=None):
        if tgt_seq is None:
            return self._generate_forward(z)
        
        else:
            return self._train_forward(z, tgt_seq, fgp)
            
    def _train_forward(self, z, tgt_seq, fgp):
        bsz, seq_len = tgt_seq.size()
        device = z.device

        mask_pos = self._generate_mask(bsz, seq_len, device)
        
        mask_tgt = tgt_seq.clone()
        mask_tgt[mask_pos] = self.mask_token_id

        pos = torch.arange(seq_len, device=device).expand(bsz, seq_len)
        embeddings = self.embedding(mask_tgt) + self.pos_embed(pos)

        context = self.norm(self.proj(z)).unsqueeze(1).expand(-1, seq_len, -1)
        # embeddings = embeddings + context

        if fgp is not None:
            fgp_emb = self.fgp_proj(fgp.float()).unsqueeze(1).expand(-1, seq_len, -1)
            # embeddings = embeddings + fgp_emb
        
        for _ in range(5):
            embeddings = self.embedding(mask_tgt) + self.pos_embed(pos) + fgp_emb + context
            output = self.encoder(embeddings.permute(1, 0, 2)).permute(1, 0, 2)
            logits = self.fc(output)
            
            predicted_tokens = torch.argmax(logits, dim=-1)
            mask_tgt = mask_tgt.clone() 
            mask_tgt[mask_pos] = predicted_tokens[mask_pos]
            


        # output = self.encoder(embeddings.permute(1, 0, 2)).permute(1, 0, 2)
        # logits = self.fc(output)

        return logits, mask_pos
    
    def _generate_forward(self, z, num_iters=5, temperature=1.2, max_len=128, method='random'):
        bsz = z.size(0)
        device = z.device
        max_len = max_len if max_len is not None else self.max_length
 
        # initialize the sequence with mask tokens
        seq = torch.full((bsz, self.max_length), self.mask_token_id, device=device)
        seq[:, 0] = self.vocab.bos_idx

        for iter_idx in range(num_iters):
            # check the mask_pos
            mask_pos = (seq == self.mask_token_id)
            if not mask_pos.any():
                break # if no mask_pos, break
            
            with torch.no_grad():
                logits = self._predict_forward(z, seq) #[B, L, V]
                probs = F.softmax(logits / temperature, dim=-1) #[B, L, V]
                
            if iter_idx == num_iters - 1:
                _, next_tokens = torch.max(probs, dim=-1)
                seq[mask_pos] = next_tokens[mask_pos]
            else:
                if method == 'random':
                    next_tokens = torch.multinomial(probs[mask_pos], 1).squeeze(-1)
                    seq[mask_pos] = next_tokens
                elif method == 'top_k':
                    next_tokens = top_k_sampling(logits, mask_pos, top_k=10, temperature=temperature)
                    seq[mask_pos] = next_tokens
                elif method == 'top_p':
                    next_tokens = top_p_sampling(logits, mask_pos, p=0.9)
                    seq[mask_pos] = next_tokens
                # elif method == 'beam':
                #     next_tokens = beam_search(probs, beam_size=3, max_len=128)
                #     seq[mask_pos] = next_tokens
            

            eos_mask = (seq == self.vocab.eos_idx)
            if iter_idx < num_iters -1:
                unfinished = ~eos_mask.any(dim=1)
                if unfinished.all():
                    break

                for idx in range(bsz):
                    if unfinished[idx]:
                        valid_pos = ~eos_mask[idx] & (seq[idx] != self.vocab.bos_idx)
                        random_mask = torch.rand(max_len, device=device) < self.mask_ratio
                        seq[idx, valid_pos] = torch.where(
                                random_mask[valid_pos],
                                torch.full_like(seq[idx, valid_pos], self.mask_token_id),
                                seq[idx, valid_pos]
                            )

            # if (seq == self.vocab.eos_idx).any(dim=1).all():
            #     break

        return seq
    
    def _predict_forward(self, z, seq):
        bsz, seq_len = seq.size()
        device = z.device

        pos = torch.arange(seq_len, device=device).expand(bsz, seq_len)
        embeddings = self.embedding(seq) + self.pos_embed(pos)

        context = self.norm(self.proj(z)).unsqueeze(1).expand(-1, seq_len, -1)
        embeddings = embeddings + context

        output = self.encoder(embeddings.permute(1, 0, 2)).permute(1, 0, 2)
        logits = self.fc(output)

        return logits
        
        
        
        
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, tgt_seq = None, fgp=None):
        
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat, mask_pos = self.decoder(z, tgt_seq, fgp)
        
        return x_hat, mask_pos, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def generate(self, x):

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
    
        return self.decoder(z, tgt_seq=None)