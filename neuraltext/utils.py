# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/11_utils.ipynb.

# %% auto 0
__all__ = ['mat2dict', 'yaml2dict', 'get_character_data_from_a_trial']

# %% ../nbs/11_utils.ipynb 4
import scipy
import yaml

# %% ../nbs/11_utils.ipynb 5
def mat2dict(
    path: str # the path to the .mat file
) -> dict: # a dictionary containing the data
    data = scipy.io.loadmat(path)
    data = {k: v for k, v in data.items() if not k.startswith("__")}
    
    return data

# %% ../nbs/11_utils.ipynb 6
def yaml2dict(path: str) -> dict:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data

# %% ../nbs/11_utils.ipynb 8
def get_character_data_from_a_trial(path: str, trial: int) -> dict:
    assert trial >= 0, "trial must be a positive integer"
    def extract_letters(x, prefix = "neuralActivityCube_"):
        return x[len(prefix):] if x.startswith(prefix) else x
    
    data = mat2dict(path)    
    trial_data = {extract_letters(key): data[key][trial] for key in data.keys() if key.startswith("neuralActivityCube_")}
    del trial_data["doNothing"]
    return trial_data
