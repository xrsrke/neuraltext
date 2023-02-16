import pytest
from neuraltext.utils import yaml2dict, get_character_data_from_a_trial

def test_yaml2dict(default_config):
    assert default_config['dataset']['n_vocabs'] == 31

def test_get_character_data_from_a_trial(default_config):
    path = "./data/Datasets/t5.2019.05.08/singleLetters.mat"
    data = get_character_data_from_a_trial(path=path, trial=2)

    assert len(data) == default_config["dataset"]["n_vocabs"]