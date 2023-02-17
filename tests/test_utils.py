from neuraltext.utils import get_character_data_from_a_trial, get_sentence_data

def test_yaml2dict(default_config):
    assert default_config['dataset']['n_vocabs'] == 31

def test_get_character_data_from_a_trial(default_config):
    path = "./data/Datasets/t5.2019.05.08/singleLetters.mat"
    data = get_character_data_from_a_trial(path=path, trial=2)

    assert len(data) == default_config["dataset"]["n_vocabs"]

def test_get_sentence_data(default_config):
    N_CHANNELS = default_config["dataset"]["n_channels"]
    data = get_sentence_data("./data/Datasets/t5.2019.05.08/sentences.mat")

    sentence, neural_data = data[0]

    assert isinstance(sentence, str)
    assert neural_data.ndim == 2
    assert neural_data.shape[-1] == N_CHANNELS