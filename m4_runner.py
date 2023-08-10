#%%

import os
import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt

from torch.optim import AdamW
import torch.nn.functional as F

from m4_exp.configurations import all_config
from m4_exp.data import get_dataloaders
from m4_exp.utils import get_hf_model, validation
from tstok.tokenizer import Tokenizer
from tstok.generic import Config
from transformers import get_scheduler




# acript args
M4_DATASET_PATH = "/shared/datasets/m4_dataset"
DATANAME = "hourly"


cfg = Config(config=all_config)
device = torch.device(cfg.training.device)


# data config 
cfg.data.max_seq_len = 256
cfg.data.tr_batch_size = 64
cfg.data.val_batch_size = 512


tokenizer   = Tokenizer(cfg.data)
dataloaders = get_dataloaders(M4_DATASET_PATH, DATANAME, tokenizer, cfg, validation=True)
tr_dl       = dataloaders['train']
te_dl       = dataloaders['test']
val_dl      = dataloaders['val']


cfg.training.grad_accu_steps = 4


model               = get_hf_model(cfg, tokenizer.vocab_size)
optimizer           = AdamW(model.parameters(), lr=cfg.training.learning_rate)
num_training_steps  = int((cfg.training.num_epochs * tr_dl.approx_num_samples() / cfg.data.tr_batch_size) // cfg.training.grad_accu_steps)
lr_scheduler        = get_scheduler(name="linear",
                                    optimizer=optimizer,
                                    num_warmup_steps=0,
                                    num_training_steps=num_training_steps)



model.train()
for iteration in range(num_training_steps):
    
    iter_s = time()
    
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(cfg.training.grad_accu_steps):
        
        batch = tr_dl.get_batch(cfg.data.tr_batch_size)
        batch = {"input_ids": batch[0].to(cfg.training.device), 'labels': batch[1].to(cfg.training.device)}
        outputs = model(**batch)
        loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)),batch['labels'].reshape(-1))
        loss = loss / cfg.training.grad_accu_steps
        loss.backward()
    
    optimizer.step()
    lr_scheduler.step()
    
    iter_e = time()

    # with torch.no_grad():
    #     wandb.log({'elapsed': iter_e - iter_s, 
    #                'lr': lr_scheduler.get_lr()[0],
    #                'loss': loss.detach().cpu().item() * cfg.ft.grad_accu_steps})

    if (iteration % cfg.io.log_interval) == 0:
        print(f"iter:{iteration}/{num_training_steps} elapsed: {iter_e - iter_s:.3f} loss:{outputs.loss.detach().item():.3f}")

    if ((iteration % cfg.io.eval_interval) == 0) and (iteration > 10):
        model.eval()
        val = validation(model, val_dl)
        te = validation(model, te_dl)

        print(f"iter:{iteration}/{num_training_steps} {val['mse']=:.3f} {val['mae']=:.3f} {te['mse']=:.3f} {te['mae']=:.3f}")

    torch.cuda.empty_cache()





#%%






