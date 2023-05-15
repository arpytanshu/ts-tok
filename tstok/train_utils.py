import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from infer import gen_forecast

@torch.no_grad()
def estimate_loss(model, dataset, ctx, cfg):
    # helps estimate an arbitrarily accurate loss over either split using many batches
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(cfg.io.eval_iters)
        for k in range(cfg.io.eval_iters):
            X, Y = dataset.get_batch(cfg.data.batch_size, split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

@torch.no_grad()
def plot_eval(model, tokenizer, device, path=None):
    def _gen_daily_signal():
        num_days = 7
        num_dp_per_day = 24
        x = np.linspace(0, num_days*2*np.pi, num_dp_per_day*num_days)
        e = np.random.randn(num_dp_per_day*num_days) * 0.3
        x = np.sin(x+e) + 5
        return x

    N = 5
    fig, axs = plt.subplots(N, 1, figsize=(20, 2*N))
    for ix, temp in enumerate(np.linspace(0.2, 0.99, N)):
        x = _gen_daily_signal()
        y = gen_forecast(x, model, tokenizer, 100, temperature=temp, top_k=100)

        axs[ix].plot(range(len(x)), y[:len(x)])
        axs[ix].plot(range(len(x), len(y)), y[len(x):])
        axs[ix].set_title(f'Temperature: {temp:.2f}')
    fig.tight_layout()
    plt.savefig(path)
    plt.show()



def get_lr(iter, cfg):
    # learning rate decay scheduler (cosine with warmup)

    # 1) linear warmup for warmup_iters steps
    if iter < cfg.training.warmup_iters:
        return cfg.training.learning_rate * iter / cfg.training.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if iter > cfg.training.lr_decay_iters:
        return cfg.training.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (iter - cfg.training.warmup_iters) / (cfg.training.lr_decay_iters - cfg.training.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return cfg.training.min_lr + coeff * (cfg.training.learning_rate - cfg.training.min_lr)


