from neuraltext.synthetic import SyntheticSentence, load_snippet

def test_create_word_snippet():
    pass

def test_load_word_snippet():
    pass


### SNIPPET
def test_load_snippet():
    data = load_snippet("a")

    # TODO: why z = (1, 13)
    assert data.shape == (1, 92)


def test_generate_synthetic_word(default_config):
    N_CHANNELS = default_config["dataset"]["n_channels"]
    N_VOCABS = default_config["dataset"]["n_vocabs"]
    sythetor = SyntheticSentence()
    sentence = "persistence is all you need"
    seq_len = len(sentence)

    neural_data, probs, signals = sythetor.generate(sentence)

    assert neural_data.shape[-1] == N_CHANNELS
    assert probs.shape == (seq_len, N_VOCABS)
    assert signals.shape == (seq_len,)

    assert probs[0].sum() == 1.0
    assert probs[-1].sum() == 1.0

    # TODO: fix, incase insert spaces
    assert signals[0] == 1
    assert signals[-1] == 0


def test_load_synthetic_word():
    pass

def test_generate_synthetic_sentence():
    pass

def test_load_synthetic_sentence():
    # load synthetic word
    # load a list of sentences
        # load a list of words
    # generate sentence
    pass