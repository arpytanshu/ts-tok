#%%

import os

from m4_exp.configurations import all_config
from m4_exp.data import(
    M4ValDataset,
    M4TrainDataLoader, 
    data_name_mapping,
    get_validataion_sequences,
    get_training_series,
    get_dataloaders
)

import pandas as pd
import matplotlib.pyplot as plt
from tstok.tokenizer import Tokenizer
from tstok.generic import Config
from torch.utils.data import DataLoader



# acript args
M4_DATASET_PATH = "/shared/datasets/m4_dataset"
DATANAME = "hourly"


cfg = Config(config=all_config)
tokenizer = Tokenizer(cfg.data)


dataloaders = get_dataloaders(M4_DATASET_PATH, DATANAME, tokenizer, cfg, validation=True)

tr_dl = dataloaders['train']
te_dl = dataloaders['test']
val_dl = dataloaders['val']







# %%

