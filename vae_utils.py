
import torch
import torch.nn.functional as F

def beam_search(logits, beam_size=3, max_len=128):
    """
    Beam search sampling.

    Args:
        logits: Logits from the model [batch_size, seq_len, vocab_size]
        beam_size: Number of beams to keep
        max_len: Maximum sequence length

    Returns:
        Generated sequences [batch_size, seq_len]
    """
    bsz, seq_len, vocab_size = logits.size()
    device = logits.device

    # Initialize beam
    sequences = torch.full((bsz, beam_size, max_len), self.pad_idx, device=device)
    sequences[:, :, 0] = self.bos_idx
    scores = torch.zeros((bsz, beam_size), device=device)

    for t in range(1, max_len):
        # Expand beam
        logits_t = logits[:, t-1, :].unsqueeze(1).expand(-1, beam_size, -1)
        probs_t = F.softmax(logits_t, dim=-1)

        # Get top-k candidates
        topk_probs, topk_tokens = torch.topk(probs_t, beam_size, dim=-1)

        # Update scores and sequences
        new_scores = scores.unsqueeze(2) + topk_probs
        new_sequences = torch.cat([sequences, topk_tokens.unsqueeze(2)], dim=2)

        # Select top-k beams
        topk_scores, topk_indices = torch.topk(new_scores.view(bsz, -1), beam_size, dim=-1)
        beam_idx = topk_indices // beam_size
        seq_idx = topk_indices % beam_size

        scores = topk_scores.view(bsz, beam_size)
        sequences = new_sequences[torch.arange(bsz).unsqueeze(1), beam_idx, :]

    # Return the best sequence for each batch
    return sequences[:, 0, :]

def top_k_sampling(logits, mask_pos, top_k=5, temperature=0.7):
    """
    Top-k sampling.

    Args:
        logits: Logits from the model [batch_size, vocab_size]
        k: Number of top tokens to consider

    Returns:
        Next tokens [batch_size]
    """
    bsz, vocab_size = logits.size()

    topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
    probs = F.softmax(topk_values/temperature, dim=-1)

    sampled_idx = torch.multinomial(probs[mask_pos],1).squeeze(-1)
    next_tokens = topk_indices[mask_pos].gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)
    return next_tokens
def top_p_sampling(logits, mask_pos, p=0.9):
    """
    Top-p (Nucleus) sampling.

    Args:
        logits: Logits from the model [batch_size, vocab_size]
        p: Probability threshold

    Returns:
        Next tokens [batch_size]
    """
    bsz, seq_len, vocab_size = logits.size()

    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    cutoff_mask = cumulative_probs > p
    sorted_probs[cutoff_mask] = 0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    sampled_idx = torch.multinomial(sorted_probs[mask_pos], 1).squeeze(-1)
    next_tokens = sorted_indices[mask_pos].gather(-1, sampled_idx.unsqueeze(-1)).squeeze(-1)
    return next_tokens