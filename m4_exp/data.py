
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tstok.tokenizer import Tokenizer
from tstok.generic import Config
from tstok.tsutils import subsequence

data_name_mapping = {
    "hourly":{
        "tr": "Hourly-train.csv",
        "te": "Hourly-test.csv",
        "m": 24,
    },
    "monthly":{
        "tr": "Monthly-train.csv",
        "te": "Monthly-train.csv",
        "m": 12,
    },
    "daily":{
        "tr": "Daily-train.csv",
        "te": "Daily-test.csv",
        "m": 1,

    },
    "quarterly":{
        "tr": "Quarterly-train.csv",
        "te": "Quarterly-train.csv",
        "m": 4,
    },
    "weekly":{
        "tr": "Weekly-train.csv",
        "te": "Weekly-train.csv",
        "m": 1,
    },
    "yearly":{
        "tr": "Yearly-train.csv",
        "te": "Yearly-train.csv",
        "m": 1,
    }    
}

class M4ValDataset(Dataset):
    def __init__(self,
                 sequences: list[np.array],
                 tokenizer: Tokenizer,
                 cfg: Config,
                 mode:str):
        # each element in series is a 1-array of time-series values
    
        self.sequences = sequences
        self.mode = mode

        self.max_seq_len = cfg.data.max_seq_len
        self.device = cfg.training.device
        self.tokenizer = tokenizer

    def __len__(self,):
        return len(self.sequences)

    def __getitem__(self, index):    
        seq = self.sequences[index]
        input_ids, params = self.tokenizer.encode(seq)
        X = torch.from_numpy(input_ids[:-1]).to(torch.long)
        Y = torch.tensor(input_ids[-1]).to(torch.long)
        return X, Y, params

class M4TrainDataLoader:
    def __init__(self,
                 series: list[np.array],
                 tokenizer: Tokenizer,
                 cfg: Config):
        # each element in series is a 1-array of time-series values
        if series is not None:
            self.series = series
            self.num_series = len(series)
            self.lengths = [len(s) for s in self.series]
            self.mode = 'train'

        self.max_seq_len = cfg.data.max_seq_len
        self.device = cfg.training.device
        self.tokenizer = tokenizer

    def _get_training_sequence(self):    
        # randomly select a series.
        series_ix = np.random.randint(self.num_series)
        # randomly select start index for a subsequence.
        ts_start_ix = np.random.randint(0, self.lengths[series_ix] - self.max_seq_len)
        # slice subsequence.
        sequence = self.series[series_ix][ts_start_ix:ts_start_ix+self.max_seq_len+1]
        return sequence
        
    def get_batch(self, batch_size):
        
        seqs = []
        for _ in range(batch_size):
            seqs.append(self._get_training_sequence())
        
        input_ids, _ = self.tokenizer.encode(seqs)
        X_ids = input_ids[:, :-1]
        Y_ids = input_ids[:, 1:]

        X_ids = torch.from_numpy(X_ids).to(torch.long)
        Y_ids = torch.from_numpy(Y_ids).to(torch.long)

        if self.device == 'cuda':
            # pin arrays x,y
            # which allows us to move them to GPU asynchronously (non_blocking=True)
            X_ids = X_ids.pin_memory().to(self. device, non_blocking=True)
            Y_ids = Y_ids.pin_memory().to(self.device, non_blocking=True)
        # else:
        #     x, y = x.to(self.device), y.to(self.device)
        return X_ids, Y_ids
    
    def approx_num_samples(self):
        return sum([len(x) - self.max_seq_len for x in self.series])

   
def get_training_series(tr_df, te_df, include_val_data=False):
    # returns a list of list of series. Each series is a list of values.
    # Each of these series will need to be subsequenced to model's max_seq_len.
    # This is done to avoid creating the attention masks.
    val_len = te_df.shape[1]
    training_series = []

    for _, tr_row in tr_df.iterrows():
        # indexing as [1:] to remove and identifier columns in data.
        tr_series = tr_row.iloc[1:].dropna().values.tolist()
        if include_val_data:
            training_series.append(tr_series)
        else:
            training_series.append(tr_series[:-val_len])
    return training_series


def get_validation_sequences(tr_df, te_df, cfg, return_sample_lengths=False, return_val=True):
    # X: A list of list of series. (batch_size x max_seq_len)
    # which is the input to the model for validation / evaluation.
    # Y: A list of targets. (batch_size x 1)
    # which is the target for the input sequence.
    te_len = te_df.shape[1]
    max_seq_len = cfg.data.max_seq_len

    all_series = []        
    for (_, tr_row), (_, te_row) in zip(tr_df.iterrows(), te_df.iterrows()):
        tr_vals = tr_row.dropna().values[1:]
        te_vals = te_row.dropna().values[1:]
        series = np.hstack([tr_vals, te_vals])
        all_series.append(series)

    test_sequences = []
    te_start_ix = - (te_len + max_seq_len) # reverse index
    te_end_ix = None # to the last
    for te_series in all_series:
        te_series = np.array(te_series)[te_start_ix : te_end_ix]
        te_subseq = subsequence(te_series, max_seq_len+1)
        test_sequences.append(te_subseq)
    test_sequences = np.vstack(test_sequences)

    return_dict = {}
    return_dict['test'] = test_sequences
    
    if return_val:
        val_sequences = []
        val_start_ix = -(te_len + te_len + max_seq_len)
        val_end_ix = -te_len
        for val_series in all_series:
            val_series = val_series[val_start_ix : val_end_ix]
            val_subseq = subsequence(val_series, max_seq_len+1)
            val_sequences.append(val_subseq)
        val_sequences = np.vstack(val_sequences)
        return_dict['val'] = val_sequences
    
    return return_dict


def get_stacked_series(base_path, data_name):

    tr_df = pd.read_csv(os.path.join(base_path, data_name_mapping[data_name]['tr']))
    te_df = pd.read_csv(os.path.join(base_path, data_name_mapping[data_name]['te']))


    all_series = []
    sample_lengths = []
    test_lengths = []
    for (_, tr_row), (_, te_row) in zip(tr_df.iterrows(), te_df.iterrows()):
        tr_vals = tr_row.dropna().values[1:]
        te_vals = te_row.dropna().values[1:]
        series = np.hstack([tr_vals, te_vals])
        all_series.append(series)
        sample_lengths.append(len(tr_vals))
        test_lengths.append(len(te_vals))
    
    return_dict = {
        'all_series':all_series,
        'sample_lengths':sample_lengths,
        'test_lengths':test_lengths
        }
    
    return return_dict

def get_dataloaders(base_path, data_name, tokenizer, cfg, validation=True):

    tr_df = pd.read_csv(os.path.join(base_path, data_name_mapping[data_name]['tr']))
    te_df = pd.read_csv(os.path.join(base_path, data_name_mapping[data_name]['te']))


    val_test_seqs = get_validation_sequences(tr_df, te_df, cfg, return_val=validation)
    training_series = get_training_series(tr_df, te_df, include_val_data=validation)
    
    return_dict = {}
    
    train_dataloader = M4TrainDataLoader(training_series, tokenizer, cfg)
    return_dict['train'] = train_dataloader
    
    test_seqs = val_test_seqs['test']
    test_dataset = M4ValDataset(test_seqs, tokenizer, cfg, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.data.val_batch_size, shuffle=False)
    return_dict['test'] = test_dataloader
    
    if validation:
        val_seqs = val_test_seqs['val']
        val_dataset = M4ValDataset(val_seqs, tokenizer, cfg, mode='val')
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.data.val_batch_size, shuffle=False)
        return_dict['val'] = val_dataloader

    return return_dict
