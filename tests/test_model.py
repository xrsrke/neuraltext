import torch
from neuraltext.model import RNN, get_vocab

def test_vocab():
    vocab = get_vocab()

    assert len(vocab) == 31
    assert vocab['a'] == 0
    assert vocab['b'] == 1

def test_rnn(default_config):
    HIDDEN_SIZE = 512
    N_CHANNELS = default_config["dataset"]["n_channels"]
    N_VOCABS = default_config["dataset"]["n_vocabs"]
    x = torch.randn(1, N_CHANNELS)

    model = RNN(
        input_size=N_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_size=N_VOCABS
    )
    character_prob, hidden, next_prob = model(x)

    assert character_prob.shape == (1, N_VOCABS)
    assert hidden.shape == (1, HIDDEN_SIZE)
    assert next_prob.shape == (1,)
    assert 0 <= next_prob.item() <= 1