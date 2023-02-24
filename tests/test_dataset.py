import torch
from torch.utils.data import DataLoader

from neuraltext.dataset import NeuralCharacterDataset, NeuralSentenceDataset
from neuraltext.utils import get_character_data_from_a_trial, get_sentence_data

def test_create_neural_character_dataset(default_config, tokenizer):
    N_CHANNELS = default_config["dataset"]["n_channels"]
    N_VOCABS = default_config["dataset"]["n_vocabs"]

    data = get_character_data_from_a_trial("./data/Datasets/t5.2019.05.08/singleLetters.mat", 0)

    dataset = NeuralCharacterDataset(data, tokenizer)
    x, y = dataset[0]

    assert len(dataset.vocabs) == default_config["dataset"]["n_vocabs"]
    assert isinstance(x, int)
    assert isinstance(y, torch.Tensor)
    assert 0 <= x <= N_VOCABS
    assert y.ndim == 1
    assert y.shape[-1] == N_CHANNELS

def test_create_dataloader_from_neural_character_dataset(tokenizer):
    data = get_character_data_from_a_trial("./data/Datasets/t5.2019.05.08/singleLetters.mat", 0)
    BATCH_SIZE = 2

    dataset = NeuralCharacterDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for _, batch in enumerate(dataloader):
        xs, ys = batch
        assert len(batch) == BATCH_SIZE
        assert len(xs) == BATCH_SIZE
        assert len(ys) == BATCH_SIZE
        break

def test_create_neural_sentence_dataset(default_config, tokenizer):
    N_CHANNELS = default_config["dataset"]["n_channels"]
    data = get_sentence_data("./data/Datasets/t5.2019.05.08/sentences.mat")

    dataset = NeuralSentenceDataset(data, tokenizer)
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 2
    assert y.shape[-1] == N_CHANNELS