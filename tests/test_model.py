import torch
from neuraltext.model import RNN

def test_rnn():
    x = torch.randn(100, 1, 1)
    model = RNN(input_size=1, hidden_size=10, output_size=1)

    y, z, h = model(x)

    assert y.shape == (1, 1, 1)
    assert z.shape == (1, 1)
    assert h.shape == (1, 1, 10)