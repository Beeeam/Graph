
def beam_search(x, beam_size=5, max_len=20):
    # Encode the input
    x = self.encoder(x)
    x = x.unsqueeze(1)
    x = x.expand(-1, beam_size, -1)
    x = x.contiguous().view(-1, x.size(2))

    # Initialize the hidden state
    hidden = self.init_hidden(x.size(0))

    # Initialize the beam
    beam = Beam(beam_size, max_len, self.vocab_size)

    # Decode the input
    for i in range(max_len):
        if beam.done():
            break

        y, hidden = self.decoder(x, hidden)
        y = y[:, -1, :]
        y = F.log_softmax(y, dim=1)

        beam.advance(y)

        if i == 0:
            hidden = hidden.expand(-1, beam_size, -1)

        idx = beam.get_current_state()
        x = self.embed(idx)
        x = x.unsqueeze(1)
        x = x.expand(-1, beam_size, -1)
        x = x.contiguous().view(-1, x.size(2))

    return beam.get_best_sequence()