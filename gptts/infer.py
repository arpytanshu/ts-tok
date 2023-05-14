
#%%

import torch

from model import GPT
from generic import Config
from tokenizer import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd

def load_stuff(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    cfg = Config(config=checkpoint['config'])
    tokenizer = Tokenizer(cfg.data)
    model = GPT(cfg.model)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, tokenizer

@torch.no_grad()
def get_forecasts(model, tokenizer, series, horizon, temperature=0.5, top_k=100):
    out = series.tolist()
    for _ in range(horizon):
        idx, params = tokenizer.encode(np.array(out).reshape(1, -1))
        idx = torch.tensor(idx).reshape(1, -1)
        generations = model.generate(idx, max_new_tokens=1, temperature=0.9, top_k=199)
        generations = generations.detach().cpu().numpy().ravel()
        y = tokenizer.decode(generations)
        y = (y * params['scale']) + params['loc']
        y = y.ravel()
        out.append(y[-1])
    return y

@torch.no_grad()
def get_batched_forecasts(model, tokenizer, b_series, horizon, temperature=0.5, top_k=100):
    ctx_len = b_series.shape[1]
    num_samples = b_series.shape[0]
    output = np.zeros((num_samples, horizon))
    output = np.hstack([b_series, output])
    for ix in range(ctx_len, ctx_len+horizon):
        ctx = output[:, ix-ctx_len:ix]
        idx, params = tokenizer.encode(ctx)
        idx = torch.tensor(idx)
        generations = model.generate(idx, max_new_tokens=1, temperature=0.9, top_k=199)
        generations = generations[:, -1].detach().cpu().numpy().ravel()
        y = tokenizer.decode(generations).reshape(-1, 1)
        y = (y * params['scale']) + params['loc']
        output[:, ix] = y.ravel()
    return output

def _gen_daily_signal():
    num_days = 7
    num_dp_per_day = 24
    x = np.linspace(0, num_days*2*np.pi, num_dp_per_day*num_days)
    e = np.random.randn(num_dp_per_day*num_days) * 0.3
    x = np.sin(x+e) + 5
    return x

def _gen_monotonic_signal():
    num_days = 7
    num_dp_per_day = 24
    x = np.linspace(0, num_days*2*np.pi, num_dp_per_day*num_days)
    e = np.random.randn(num_dp_per_day*num_days) * 0.3
    x = x+e + 5
    return x


M = 2
N = 3
B = M * N
T = 256
F = 100


files = glob("/Users/arpitanshulnu/Documents/checkout/arpytanshu/gpt-ts-data/*/*.csv")
checkpoint_path = '../output/test_run_4/test_run_4-ckpt650.pt'
model, tokenizer = load_stuff(checkpoint_path)
model.eval()
s

b_series = []
targets = []
for ix in range(M*N):
    while True:
        df = pd.read_csv(np.random.choice(files))
        if len(df) > T+F+100: break;
    L = len(df)
    s_ix = np.random.randint(0, L-T-F)
    series = df.value.values[s_ix:s_ix+T]
    orig = df.value.values[s_ix+T:s_ix+T+F]
    b_series.append(series)
    targets.append(orig)

b_series = np.stack(b_series)
targets = np.stack(targets)

forecasts = get_batched_forecasts(model, tokenizer, b_series, horizon=F, temperature=1.1)


fig, axs = plt.subplots(N, M, figsize=(6*M, 2*N))
axs = axs.ravel()
for ix in range(N*M):
    ax = axs[ix]
    series = forecasts[ix]
    ax.plot(range(T), series[:T], c='b', label='orig')
    # ax.plot(range(T, T+F), targets[ix], label='target', c='b', linestyle='--', alpha=0.3)
    ax.plot(range(T, T+F), series[T:T+F], label='forecast', c='orange')

    # remove axes borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.legend

fig.tight_layout()
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower left')
plt.show()

