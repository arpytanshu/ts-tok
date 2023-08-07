import time


data_config = {
    # dataloader
    "max_seq_len": 256, # 512 + 1 for the target
    
    # if gradient_accumulation_steps > 1, this is the micro-batch size
    # This is same as the max_seq_len in data_config
    "batch_size": 64,
    # tokenizer
    "bin_size": 0.005,
    "max_coverage": .9998,
    "train_ratio": 0.95
}


model_config = {
    "n_layer": 6,
    "n_embd": 128,
    "n_head": 4,
    "vocab_size": None, # filled by tokenizer
    "block_size": 512, 
    "bias": False,
    "dropout": 0.1,
}

# only training specific configs
training_config = {
    # adamw optimizer
    "learning_rate": 6e-4,              # max learning rate
    "max_iters": 600000,                # total number of training iterations
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,                   # clip gradients at this value, or disable if == 0.0
    
    # learning rate decay settings
    "decay_lr": True,                   # whether to decay the learning rate
    "warmup_iters": 200,                # how many steps to warm up for
    "lr_decay_iters": 600000,           # should be ~= max_iters per Chinchilla
    "min_lr": 6e-5, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # training opts
    "grad_accu_steps": 5 * 8, # used to simulate larger batch sizes
    
    # system - training opts
    "device": "cpu", # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    "dtype": "float32", # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    "compile": False, # use PyTorch 2.0 to compile the model to be faster
}

# model save-load, wandb, loss-progress, distributed training etc.
logistics_config = {
    # wandb
    "wandb_log": True,
    "wandb_project": "gptts",
    "wandb_run_name": "gptts_run_" + time.strftime('%Y%B%d_%X'),

    # I/O
    "out_dir": None, # filled by the script
    "eval_interval": 20,
    "log_interval": 1,
    "eval_iters": 2,                # was 200 in default.
    "eval_only": False ,            # if True, script exits right after the first eval
    "always_save_checkpoint": False, # if True, always save a checkpoint after each eval
    "init_from": 'scratch',         # 'scratch' or 'resume'

    # ddp settings
    "backend": 'nccl', # 'nccl', 'gloo', etc.
    "preprocesses_data_path": '../processed_data/',
}

all_config = {
    "data": data_config,
    "model": model_config,
    "training": training_config,
    "io": logistics_config,
}
