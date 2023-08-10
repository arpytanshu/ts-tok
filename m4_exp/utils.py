import numpy as np
import torch
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
    smape = (2 * 100 * num) / (horizon * den)
    return smape

def mae(y, y_hat):
    return np.abs(y - y_hat).mean()

def mse(y, y_hat):
    return np.square(y - y_hat).mean()


def validation(model, dataloader):
    model.eval()
    tokenizer = dataloader.dataset.tokenizer
    
    forecasts = []
    targets = []

    for batch in dataloader:
        val_x, val_y, params = batch
        val_x = val_x.to(model.device)
        params = {k:params[k].numpy() for k in params.keys()}
        
        with torch.no_grad():
            out = model(val_x)
            probs = torch.softmax(out.logits[:, -1, :], dim=1).cpu()
        

        ids = torch.multinomial(probs, num_samples = 1)
        del(out)
        
        val_y = tokenizer.decode(val_y, params).squeeze()
        val_y_hat = tokenizer.decode(ids.detach().cpu().numpy(), params).squeeze()

        forecasts.append(val_y_hat)
        targets.append(val_y)

        print('|', end='')

        torch.cuda.empty_cache()

    Y_hat = np.hstack(forecasts)
    Y = np.hstack(targets)

    metric_mse = mse(Y, Y_hat)
    metric_mae = mae(Y, Y_hat)
    print()

    return {'mae': metric_mae, 'mse': metric_mse}

    
