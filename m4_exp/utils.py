import torch
import torch.nn.functional as F
import numpy as np
from time import time
from transformers import LlamaConfig, LlamaForCausalLM


def get_hf_model(cfg, vocab_size):
    model_config = LlamaConfig( 
        vocab_size=vocab_size,
        hidden_size=cfg.model.hidden_size,
        intermediate_size=cfg.model.intermediate_size,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        max_position_embeddings=cfg.data.max_seq_len,
    )
    device = torch.device(cfg.training.device)
    model = LlamaForCausalLM(model_config).to(torch.bfloat16).to(device)
    return model

def get_num_params(model):
    count = 0
    for p in model.parameters():
        count += p.numel()
    return count

def sMAPE(y, y_hat, horizon):    
    num = np.abs(y - y_hat)
    den = np.abs(y) + np.abs(y_hat)
    smape = (2 * (num / den).sum() * 100) / horizon
    return smape

def MASE(context, y, y_hat, horizon, m):
    # context: context data used for generation
    # horizon: num data points forecasted.
    # m: period of data, eg: 24 for hourly, 12 for monthly
    n = len(context)
    num = np.abs(y - y_hat).sum()
    
    den_y = context[m:]
    den_y_shifted = context[:-m]
    den_coeff = 1/(n-horizon)
    den = den_coeff * np.abs(den_y - den_y_shifted).sum()
    
    mase = num / (horizon * den)
    return mase


def mae(y, y_hat):
    return np.abs(y - y_hat).mean()

def mse(y, y_hat):
    return np.square(y - y_hat).mean()


def validation(model, dataloader):
    '''
    This validation uses a generic dataloader, that stack all test samples
    together and reports mse/mae/ce_loss over them.
    For reporting MASE, the number of avl datapoints shall also be considered,
    and has been implemented in validation 2, which does not use a  dataloader.
    '''
    model.eval()
    tokenizer = dataloader.dataset.tokenizer

    forecasts = []
    targets = []
    elapsed = []
    losses = []

    for batch in dataloader:
        tick = time()

        val_x, val_y, params = batch
        val_x = val_x.to(model.device)
        val_y = val_y.to(model.device)
        params = {k:params[k].numpy() for k in params.keys()}
        
        with torch.no_grad():
            out = model(val_x, output_hidden_states=False, output_attentions=False, use_cache=False)
            loss = F.cross_entropy(out.logits[:, -1, :], val_y)
            probs = torch.softmax(out.logits[:, -1, :], dim=1).cpu()

        ids = torch.multinomial(probs, num_samples = 1)
        val_y = tokenizer.decode(val_y.cpu().numpy(), params).squeeze()
        val_y_hat = tokenizer.decode(ids.detach().cpu().numpy(), params).squeeze()

        losses.append(loss.detach().cpu().item())
        forecasts.append(val_y_hat)
        targets.append(val_y)

        print('|', end='')
        torch.cuda.empty_cache()

        elapsed.append((time() - tick) / val_x.shape[0])
    print()

    Y_hat = np.hstack(forecasts)
    Y = np.hstack(targets)

    mean_ce_loss = sum(losses) / len(losses)
    mean_time_per_sample = sum(elapsed) / len(elapsed)
    metric_mse = mse(Y, Y_hat)
    metric_mae = mae(Y, Y_hat)

    return {'mae': metric_mae,
            'mse': metric_mse,
            'time_per_sample':mean_time_per_sample,
            'mean_ce_loss': mean_ce_loss}

    
