import pytest
from transformers import AutoTokenizer

from neuraltext.utils import yaml2dict

@pytest.fixture
def default_config():
    return yaml2dict('./configs/default.yaml')

@pytest.fixture
def tokenizer(default_config):
    return AutoTokenizer.from_pretrained(default_config['tokenizer']['name'])