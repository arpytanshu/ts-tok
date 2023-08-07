#%%

import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import argparse
import os
import pickle
import matplotlib.pyplot as plt

from tstok.tokenizer import Tokenizer
from tstok.generic import Config, progress_bar
from tstok.tsutils import subsequence
from custom_exp.configurations import data_config


def norm_and_tokenize(seq, tokenizer):
    Xs = seq[:, :-1]
    Ys = seq[:, 1:]

    means = np.zeros(Xs.shape)
    std = np.zeros(Ys.shape)
    for i in range(Xs.shape[1]):
        means[:, i] = Xs[:, :i+1].mean(axis=1)
        std[:, i] = Xs[:, :i+1].std(axis=1)

    Xs_std = (Xs - Xs.mean(axis=1).reshape(-1, 1)) / (Xs.std(axis=1).reshape(-1, 1) + 1e-6)
    Ys_std = (Ys - means) / (std + 1e-6)

    X_clip = tokenizer.clip(Xs_std, tokenizer.bins[0], tokenizer.bins[-1])
    Y_clip = tokenizer.clip(Ys_std, tokenizer.bins[0], tokenizer.bins[-1])

    X_ids = np.clip(np.digitize(X_clip, tokenizer.bins, right=False)-1, 0, len(tokenizer.bins)-2)
    Y_ids = np.clip(np.digitize(Y_clip, tokenizer.bins, right=False)-1, 0, len(tokenizer.bins)-2)

    return X_ids, Y_ids


def get_stacked_sequences(files, tokenizer, cfg):
    # for each file, create subsequences and tokenize 

    stacked_Xs = []
    stacked_Ys = []
    for ix, file in enumerate(files):
        df = pd.read_csv(file)

        abort_flag1 = len(df) < (cfg.max_seq_len+1)
        abort_flag2 = df.value.nunique() == 1
        if abort_flag1 or abort_flag2:
            continue;

        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)
        seq = subsequence(df.value.values, cfg.max_seq_len+1)
        
        X_ids, Y_ids = norm_and_tokenize(seq, tokenizer)

        stacked_Xs.append(X_ids)
        stacked_Ys.append(Y_ids)

        progress_bar(ix, len(files), bar_length=20)

    stacked_Xs = np.vstack(stacked_Xs).astype(np.int16)
    stacked_Ys = np.vstack(stacked_Ys).astype(np.int16)

    return {"X": stacked_Xs, "Y": stacked_Ys}


def do_stuff(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    out_path = Path(args.out_path)
    
    # setup tokenizer
    overrides = {
        "bin_size": args.bin_size, 
        "max_seq_len": args.max_seq_len
        }
    cfg = Config(config=data_config|overrides)
    tokenizer = Tokenizer(cfg)
    print(f"tokenizer created w/ config: {{bin_size:{tokenizer.BIN_SIZE}, max_coverage:{tokenizer.MAX_COVERAGE}}}")
    print(f"number of tokens(bins) created: {tokenizer.vocab_size}")


    all_files = glob(args.base_path)[:]
    np.random.shuffle(all_files)
    split_ix = int(args.train_ratio * len(all_files))

    meta = {'vocab_size': tokenizer.vocab_size, 'max_seq_len': args.max_seq_len, 'bin_size': args.bin_size}
    with open(out_path/'meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    val_files = all_files[split_ix:]
    print(f"\nnum val files: {len(val_files)}")
    val_sequences = get_stacked_sequences(val_files, tokenizer, cfg)
    with open(out_path/'val.bin', 'wb') as f:
        np.save(f, val_sequences)

    train_files = all_files[:split_ix]
    print(f"\nnum train files: {len(train_files)}")
    train_sequences = get_stacked_sequences(train_files, tokenizer, cfg)
    with open(out_path/'train.bin', 'wb') as f:
        np.save(f, train_sequences)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--train_ratio", type=float, default=0.95)
    parser.add_argument("--bin_size", type=float, default=0.005)
    parser.add_argument("--max_seq_len", type=int, default=512)
    args = parser.parse_args()

    do_stuff(args)



'''
args = {'base_path': "/Users/arpitanshulnu/Documents/checkout/arpytanshu/gpt-ts-data/*/*.csv",
        'out_path': "processed_data",
        'train_ratio': 0.95,
        'bin_size': 0.005,
        'max_seq_len': 256,
        'out_path': "test"}
args = Config(config=args)
# do_stuff(args)

python prepare_custom_data.py --base_path "/Users/arpitanshulnu/Documents/checkout/arpytanshu/gpt-ts-data/*/*.csv" --out_path ./processed_data --train_ratio 0.95
'''
