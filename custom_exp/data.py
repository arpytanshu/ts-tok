
import os
import torch
import numpy as np
import pandas as pd
from tstok.tokenizer import Tokenizer
from tstok.generic import Config

class CustomDataset:
    def __init__(self,
                 series: list[np.array],
                 tokenizer: Tokenizer,
                 cfg: Config):
        # each element in series is a 1-array of time-series values
        self.series = series
        self.tr_series = series[:int(len(series) * cfg.data.train_ratio)]
        self.val_series = series[int(len(series) * cfg.data.train_ratio):]

        self.tr_lengths = [len(s) for s in self.tr_series]
        self.val_lengths = [len(s) for s in self.val_series]
        
        self.max_seq_len = cfg.data.max_seq_len
        self.train_ratio = cfg.data.train_ratio
        self.device = cfg.training.device
        self.tokenizer = tokenizer

    def _get_sample(self, split):
        if split == 'train':
            # randomly select a series
            series_ix = np.random.randint(len(self.tr_series))
            # randomly select a subsequence
            ts_start_ix = np.random.randint(0, self.tr_lengths[series_ix] - self.max_seq_len)
        else:
            series_ix = np.random.randint(len(self.val_series))
            ts_start_ix = np.random.randint(0, self.val_lengths[series_ix] - self.max_seq_len)

        sequence = self.series[series_ix][ts_start_ix:ts_start_ix+self.max_seq_len+1]
        return sequence[:-1], sequence[1:]
    
    def _norm_and_tokenize(self, Xs, Ys):
        '''
        Xs.shape => batch x seq_len
        Ys.shape => batch x seq_len
        '''
        # compute params that will be used as standardizing param for successive targets.
        hindsight_means = np.zeros(Xs.shape)
        hindsight_std = np.zeros(Ys.shape)
        for i in range(Xs.shape[1]):
            hindsight_means[:, i] = Xs[:, :i+1].mean(axis=1)
            hindsight_std[:, i] = Xs[:, :i+1].std(axis=1)

        # standardize the context windows using it's own mean and std
        Xs_std = (Xs - Xs.mean(axis=1).reshape(-1, 1)) / (Xs.std(axis=1).reshape(-1, 1) + 1e-6)

        # standardize the targets using the context windows' lagging mean and std
        Ys_std = (Ys - hindsight_means) / (hindsight_std + 1e-6)
        
        X_ids = self.tokenizer.digitize(Xs_std)
        Y_ids = self.tokenizer.digitize(Ys_std)

        return X_ids, Y_ids

    def get_batch(self, batch_size, split):
        Xs = []; Ys = []
        for _ in range(batch_size):
            X, Y = self._get_sample(split)
            Xs.append(X)
            Ys.append(Y)
        
        Xs = np.stack(Xs); Ys = np.stack(Ys)
        X_ids, Y_ids = self._norm_and_tokenize(Xs, Ys)

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




def get_custom_dataset(base_path, data_name, tokenizer, cfg):
    # data_dir: directory containing csv files
    # tokenizer: tokenizer object
    # max_seq_len: maximum length of the sequence
    data_name_mapping = {
        'weather': 'weather/weather.csv',
        'electricity': 'electricity/electricity.csv',
        'exchange_rate': 'exchange_rate/exchange_rate.csv',
        'traffic': 'traffic/traffic.csv',
        'illness': 'illness/national_illness.csv',
        'ettm1': 'ETT-small/ETTm1.csv',
        'ettm2': 'ETT-small/ETTm2.csv',
        'etth1': 'ETT-small/ETTh1.csv',
        'etth2': 'ETT-small/ETTh2.csv'
    }
    file = os.path.join(base_path, data_name_mapping[data_name])
    df = pd.read_csv(file)

    df = pd.read_csv(os.path.join(base_path, data_name_mapping[data_name]))
    series = [df[cols].values for cols in set(list(df.columns))-{'date'}]
    dataset = CustomDataset(series, tokenizer, cfg)

    return dataset