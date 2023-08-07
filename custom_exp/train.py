

#%%

import os
import time
import torch
import shutil
from contextlib import nullcontext


from model import GPT
from tstok.generic import Config
from configurations import all_config
from tstok.tokenizer import Tokenizer
from train_utils import estimate_loss, plot_eval, get_lr #, CustomDataset
from data import CustomDataset, get_custom_dataset


SEED_OFFSET = 0
DATA_BASE_PATH = '/shared/datasets/all_six_datasets/all_six_datasets/' # script arg
OUT_DIR = 'temp/' # script arg
PERSIST_DIR_BASE = "/content/drive/MyDrive/TEMP/gptts-chkpts/"

offload_to_drive = False
eval_plot = True

DATASET_NAME = 'ettm1'

cfg = Config(config=all_config)




# Mandatory configs to override =======|
cfg.io.wandb_log = False
cfg.io.wandb_project = 'gptts'
cfg.io.out_dir = OUT_DIR
#======================================|


# other configs to override ===========|
cfg.io.init_from = 'scratch'
cfg.io.eval_interval = 20
cfg.io.eval_iters = 5
cfg.training.device = "cuda" if torch.cuda.is_available() else 'cpu' #'cuda:0'
cfg.training.grad_accu_steps = 1
cfg.training.learning_rate = 5e-3
cfg.data.batch_size = 512
#======================================|


# model condfigs to override ==========|
cfg.model.n_embd = 256
cfg.model.n_head = 8
cfg.model.block_size = 256
cfg.model.n_layer = 8
cfg.model.dropout = 0.05
#======================================|


# if init_from='resume', these values are loaded from checkpoint
iter_num = 0
best_val_loss = 1e9


# just chores
# ==========================================|
os.makedirs(cfg.io.out_dir, exist_ok=True)
torch.manual_seed(1337 + SEED_OFFSET)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in cfg.training.device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[cfg.training.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokens_per_iter = cfg.training.grad_accu_steps * cfg.data.batch_size * cfg.model.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")


# ==========================================|




if cfg.io.init_from == 'scratch':
    cfg.io.wandb_run_name = "gptts_run_" + time.strftime('%Y%B%d_%X')
    print('init model using following config:\n', cfg.model)    
    tokenizer = Tokenizer(cfg.data)
    cfg.model.vocab_size = tokenizer.vocab_size
    model = GPT(cfg.model)

elif cfg.io.init_from == 'resume':
    print(f"Resuming training from {cfg.io.out_dir}")
    ckpt_path = os.path.join(cfg.io.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=cfg.training.device)
    cfg.io.wandb_run_name = checkpoint['wandb_run_name']
    checkpoint_model_args = checkpoint['model_args']

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        cfg.model.__dict__[k] = checkpoint_model_args[k]
    # create the model
    tokenizer = Tokenizer(Config(config=checkpoint['config']['data']))
    model = GPT(cfg.model)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']



# crop down the model block size if desired, using model surgery
if cfg.model.block_size < model.config.block_size:
    model.crop_block_size(cfg.model.block_size)
    cfg.model.block_size = cfg.model.block_size # so that the checkpoint will have the right value
model.to(cfg.training.device)


dataset = get_custom_dataset(DATA_BASE_PATH, DATASET_NAME, tokenizer, cfg)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(cfg.training.dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(cfg.training.weight_decay,
                                       cfg.training.learning_rate, 
                                       (cfg.training.beta1, cfg.training.beta2),
                                       device_type)

if cfg.io.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if cfg.training.compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# logging
if cfg.io.wandb_log:
    import wandb
    wandb.init(project=cfg.io.wandb_project, name=cfg.io.wandb_run_name)

# training loop
X, Y = dataset.get_batch(cfg.data.batch_size, 'train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # unwrap DDP container if needed
running_mfu = -1.0


#%%

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num, cfg) if cfg.training.decay_lr else cfg.training.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % cfg.io.eval_interval == 0:
        losses = estimate_loss(model, dataset, ctx, cfg)
        if eval_plot:
            plot_eval(model, tokenizer, cfg.training.device, path=os.path.join(cfg.io.out_dir, str(iter_num)))
        
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if cfg.io.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentagex
            })
        if losses['val'] < best_val_loss or cfg.io.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': cfg.model._get_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': cfg._get_dict(),
                    'wandb_run_name': cfg.io.wandb_run_name,
                }
                print(f"saving checkpoint to {cfg.io.out_dir}")
                torch.save(checkpoint, os.path.join(cfg.io.out_dir, 'ckpt.pt'))
                if offload_to_drive:
                    src = os.path.join(cfg.io.out_dir, 'ckpt.pt')
                    dst = os.path.join(PERSIST_DIR_BASE, 'ckpt.pt')
                    shutil.copy(src, dst)
                    print('copied checkpoint to drive.')
    if iter_num == 0 and cfg.io.eval_only:
        break;

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(cfg.training.grad_accu_steps):

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / cfg.training.grad_accu_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = dataset.get_batch(cfg.data.batch_size, 'train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if cfg.training.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % cfg.io.log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * cfg.training.grad_accu_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(cfg.data.batch_size * cfg.training.grad_accu_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > cfg.training.max_iters:
        break


    # %%
