#%%

import os
import torch
import numpy as np
from time import time, strftime
import matplotlib.pyplot as plt

from torch.optim import AdamW
import torch.nn.functional as F

from m4_exp.configurations import all_config
from m4_exp.data import get_dataloaders
from m4_exp.utils import get_hf_model, validation
from tstok.tokenizer import Tokenizer
from tstok.generic import Config
from transformers import get_scheduler




# script args
M4_DATASET_PATH = "/shared/datasets/m4_dataset"
DATANAME = "hourly"
CHECKPOINT_BASE_PATH = "/shared/CO/arpytanshu_/ts-tok/checkpoints/"
EXPERIMENT_NAME = 'dev7'




# setup configurations
# ####################
cfg = Config(config=all_config)

# not hotplugs - change must init a new model.
cfg.data.max_seq_len = 512

# hotpluggable configs - change work across resumes.
cfg.data.tr_batch_size = 96
cfg.data.val_batch_size = 512
cfg.training.grad_accu_steps = 4
cfg.training.learning_rate = 6e-4
cfg.training.grad_checkpointing = False



# mandatory things - These objects will be created - as is - in all flows.
# #######################################################################
checkpoint_path     = os.path.join(CHECKPOINT_BASE_PATH,
                               EXPERIMENT_NAME,
                               "chkpt.pt")
device              = torch.device(cfg.training.device)
tokenizer           = Tokenizer(cfg.data)
dataloaders         = get_dataloaders(M4_DATASET_PATH,
                                      DATANAME,
                                      tokenizer,
                                      cfg,
                                      validation=False)
tr_dl               = dataloaders['train']
te_dl               = dataloaders['test']



# Init state - objects  to create  when initializing from scratch
# ###############################################################
def init_stuff(cfg):
    model               = get_hf_model(cfg, tokenizer.vocab_size)
    
    optimizer           = AdamW(model.parameters(),
                                lr=cfg.training.learning_rate, 
                                betas=(cfg.training.beta1, cfg.training.beta2))
    num_training_steps  = int((cfg.training.num_epochs * tr_dl.approx_num_samples() /\
                                cfg.data.tr_batch_size) // cfg.training.grad_accu_steps)
    lr_scheduler        = get_scheduler(name="cosine",
                                        optimizer=optimizer,
                                        num_warmup_steps=0,
                                        num_training_steps=num_training_steps)

    if cfg.training.grad_checkpointing:
        model.gradient_checkpointing_enable()
        assert(model.model.gradient_checkpointing)
        assert(model.training)

    chkpt_dir = os.path.dirname(checkpoint_path)
    os.makedirs(chkpt_dir)
    print(f"Created checkpoint directory at {chkpt_dir=}")

    return {
        'model': model,
        'optimizer': optimizer,
        'num_training_steps': num_training_steps,
        'lr_scheduler':  lr_scheduler,
        }

def checkpoint_stuff(path, **kwargs):
    '''
    use like this:
    # chkpt_keys = checkpoint_stuff(path, a=1, b=2, c=3)
    # chkpt_keys = checkpoint_stuff(path, **{'d':4, 'e':5})
    '''
    if not os.path.exists(os.path.dirname(path)):
        raise Exception("Parent directory of provided checkpoint path does not exist.")

    print(f"Checkpointing the following keys at {path=}:")
    checkpoint_keys = list(kwargs.keys())
    
    for k in checkpoint_keys:
        print(k, ',', end=' ')
    print()

    torch.save(kwargs, path)
    return checkpoint_keys






# create new everything
if not os.path.exists(checkpoint_path):
    print("Initializing model & optimizer.")
    stuff               = init_stuff(cfg)
    
    model               = stuff['model']
    optimizer           = stuff['optimizer']
    num_training_steps  = stuff['num_training_steps']
    lr_scheduler        = stuff['lr_scheduler']
    cfg                 = cfg
    iter_num            = 0
    best_val_scores     = {'ce': 1e7, 'mae': 1e7}

    
    cfg.io.wandb_run_name = strftime("%Y%m%d_%H%M")

# load from checkpoint
else:
    print(f"A checkpoint already exists at {checkpoint_path=}")
    print(f"Attempting load...")
    
    chkpt               = torch.load(checkpoint_path)

    num_training_steps  = chkpt['num_training_steps']
    iter_num            = chkpt['iter_num']
    best_val_scores     = chkpt['best_val_loss']
    cfg.load_state_dict(chkpt['config'])
    model               = get_hf_model(cfg, tokenizer.vocab_size)
    model.load_state_dict(chkpt['model'])
    optimizer           = AdamW(model.parameters())
    optimizer.load_state_dict(chkpt['optimizer'])
    lr_scheduler        = get_scheduler(name="cosine",
                                    optimizer=optimizer,
                                    num_warmup_steps=0,
                                    num_training_steps=num_training_steps)
    lr_scheduler.load_state_dict(chkpt['lr_scheduler'])
    lr_scheduler.step()
    print(f"Load successful. Resuming train loop...")


if cfg.io.wandb_log:
    import wandb
    wandb.init(project=cfg.io.wandb_project, name=cfg.io.wandb_run_name)


for iteration in range(iter_num, num_training_steps):
    model.train()    
    iter_s = time()
    
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(cfg.training.grad_accu_steps):
        
        batch = tr_dl.get_batch(cfg.data.tr_batch_size)
        batch = {"input_ids": batch[0].to(cfg.training.device), 
                 'labels': batch[1].to(cfg.training.device)}
        outputs = model(**batch)
        loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)),
                               batch['labels'].reshape(-1))
        loss = loss / cfg.training.grad_accu_steps
        loss.backward()
    
    optimizer.step()
    lr_scheduler.step()
    
    iter_e = time()

    train_loss_to_log = loss.detach().cpu().item() * cfg.training.grad_accu_steps

    if (iteration % cfg.io.log_interval) == 0:
        print(f"iter:{iteration}/{num_training_steps} elapsed: {iter_e - iter_s:.3f} loss:{train_loss_to_log:.3f}")

    if ((iteration % cfg.io.eval_interval) == 0) and (iteration > 10):
        model.eval()
        te = validation(model, te_dl)
        print(f"iter:{iteration}/{num_training_steps} {te['mse']=:.3f} {te['mae']=:.3f} {te['mean_ce_loss']=:.3f}")

        if cfg.io.wandb_log:
            wandb.log(step=iteration,
                      data={
                          'val_ce_loss': round(te['mean_ce_loss'], 3),
                          'val_mae': round(te['mae'], 3)})
        
        if (te['mean_ce_loss'] < best_val_scores['ce']) or (te['mae'] < best_val_scores['mae']):
            best_val_scores = {'ce': te['mean_ce_loss'], 'mae': te['mae']}
            
            # save things to checkpoint
            checkpoint_stuff(path=checkpoint_path,
                            model=model.state_dict(),
                            optimizer=optimizer.state_dict(),
                            config=cfg.state_dict(),
                            num_training_steps=num_training_steps,
                            lr_scheduler=lr_scheduler.state_dict(),
                            iter_num=iteration,
                            best_val_loss=best_val_scores)


    if cfg.io.wandb_log:
        wandb.log(step=iteration,
                  data={
                      'elapsed': iter_e - iter_s, 
                      'lr': lr_scheduler.get_lr()[0],
                      'tr_ce_loss': train_loss_to_log})
        
            

    # torch.cuda.empty_cache()

# %%
