import torch
from neuraltext.model import RNN, get_vocab

def test_vocab():
    vocab = get_vocab()

    assert len(vocab) == 31
    assert vocab['a'] == 0
    assert vocab['b'] == 1

def test_rnn():
    x = torch.randn(100, 1, 1)
    model = RNN(input_size=1, hidden_size=10, output_size=1)

    y, z, h = model(x)

    assert y.shape == (1, 1, 1)
    assert z.shape == (1, 1)
    assert h.shape == (1, 1, 10)
    assert 0 <= z.item() <= 1