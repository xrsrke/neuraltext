from torch.utils.data import DataLoader
from neuraltext.dataset import NeuralCharacterDataset, NeuralSentenceDataset
from neuraltext.utils import get_character_data_from_a_trial

def test_create_neural_character_dataset(default_config):
    data = get_character_data_from_a_trial("./data/Datasets/t5.2019.05.08/singleLetters.mat", 0)
    dataset = NeuralCharacterDataset(data)
    assert len(dataset.vocabs) == default_config["dataset"]["n_vocabs"]

def test_create_dataloader_from_neural_character_dataset():
    data = get_character_data_from_a_trial("./data/Datasets/t5.2019.05.08/singleLetters.mat", 0)
    BATCH_SIZE = 2

    dataset = NeuralCharacterDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for _, batch in enumerate(dataloader):
        xs, ys = batch
        assert len(batch) == BATCH_SIZE
        assert len(xs) == BATCH_SIZE
        assert len(ys) == BATCH_SIZE
        break

def test_create_neural_sentence_dataset():
    pass