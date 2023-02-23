import torch
from neuraltext.model import RNN, get_vocabs

def test_vocab():
    vocab = get_vocabs()

    assert len(vocab) == 31
    assert vocab['a'] == 0
    assert vocab['b'] == 1

def test_rnn_forward_pass_without_hidden(default_config):
    HIDDEN_SIZE = 512
    N_CHANNELS = default_config["dataset"]["n_channels"]
    N_VOCABS = default_config["dataset"]["n_vocabs"]
    BATCH_SIZE = 1

    x = torch.randn(BATCH_SIZE, N_CHANNELS)

    model = RNN(
        input_size=N_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_size=N_VOCABS
    )
    character_probs, hidden, next_prob = model(x, hidden=None)

    assert character_probs.shape == (BATCH_SIZE, N_VOCABS)
    assert hidden.shape == (BATCH_SIZE, HIDDEN_SIZE)
    assert next_prob.shape == (BATCH_SIZE,)

    assert torch.allclose(torch.sum(character_probs, dim=-1), torch.tensor(1.0))
    assert torch.all(character_probs >= 0), "Values must be non-negative"
    assert 0 <= next_prob[0] <= 1