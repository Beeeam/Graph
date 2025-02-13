import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from copy import deepcopy

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
            nn.Linear(hidden_size, latent_dim))
        
        self.log_var_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_dim))

    def forward(self, data):

        x = self.pretrained_model(data)

        mu = self.mu_layer(x)
        mu = torch.tanh(mu)
        log_var = self.log_var_layer(x)
        log_var = torch.clamp(log_var, min=-4, max=0)
        return mu, log_var
    
class SMILESDecoder(nn.Module):
    def __init__(self, vocab, max_length=128, latent_dim=128, embed_dim=128, dim_feedforward = 512, nhead=4, 
                 num_layers=2):
        super(SMILESDecoder, self).__init__()
        
        self.vocab = vocab
        self.max_length = max_length
        self.vocab_size = len(vocab)
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.hidden_vec = None

        self.embedding = nn.Embedding(self.vocab_size, embed_dim, padding_idx = vocab.pad_idx)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward = dim_feedforward)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, self.vocab_size)

        self.proj = nn.Linear(latent_dim, embed_dim)
        
    def forward(self, z, tgt_seq = None):
        bsz = z.size(0)
        device = z.device

        hidden_state = self.proj(z).unsqueeze(0) # (1, batch_size, embed_dim)

        if tgt_seq is None:
            logits_list, states_list = [], []
            input_ids = torch.full((bsz, 1), self.vocab.bos_idx, device=device)

            for i in range(self.max_length):
                # pos = torch.tensor([i], device=device).expand(bsz)
                current_seq_len = input_ids.size(1)
                pos = torch.arange(current_seq_len, device=device).expand(bsz, -1)
                pos_emb = self.pos_embed(pos)

                tgt_emb = self.embedding(input_ids) + pos_emb
                tgt = tgt_emb.permute(1, 0, 2)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_seq_len).to(device)

                output = self.decoder(tgt, memory=hidden_state, tgt_mask=tgt_mask)
                # memory_key_padding_mask=None, past_key_values=cache)

                # logits = self.fc(output.squeeze(0))
                logits = self.fc(output[-1])  # [bsz, vocab_size]
                next_tokens = logits.argmax(-1, keepdim=True)  # [bsz, 1]

                logits_list.append(logits.unsqueeze(1))
                states_list.append(output[-1].unsqueeze(1))

                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                if (next_tokens == self.vocab.eos_idx).all():
                    break

            gen_len = len(logits_list)
            padded_logits = torch.zeros(bsz, self.max_length, self.vocab_size, device=device)
            padded_states = torch.zeros(bsz, self.max_length, self.embed_dim, device=device)
            
            for i in range(gen_len):
                padded_logits[:, i, :] = logits_list[i].squeeze(1)
                padded_states[:, i, :] = states_list[i].squeeze(1)

            self.hidden_vec = padded_states
            return padded_logits
        
        else:
            seq_len = tgt_seq.size(1)
            pos = torch.arange(seq_len, device=device).expand(bsz, seq_len)
            tgt_emb = self.embedding(tgt_seq) + self.pos_embed(pos)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            output = self.decoder(tgt_emb.permute(1, 0, 2), hidden_state, tgt_mask=tgt_mask)
            
            logits = self.fc(output.permute(1, 0, 2))
            self.hidden_vec = output.permute(1, 0, 2)
            
            return logits

    
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x, tgt_seq = None):
        
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, tgt_seq)
        
        return x_hat, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def generate(self, x):

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
    
        return self.decoder(z, tgt_seq=None)