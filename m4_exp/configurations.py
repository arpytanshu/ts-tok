import time


data_config = {
    # dataloader
    "max_seq_len": 256,
    "tr_batch_size": 64,
    "val_batch_size": 512,
    # tokenizer config
    "bin_size": 0.005,
    "max_coverage": .9998,
    
}


model_config = {
    "vocab_size": None,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "max_position_embeddings": None
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
    "num_epochs": 3,
    "grad_accu_steps": 1,
    "grad_checkpointing": False,
    
    # system - training opts
    "device": "cuda:0", # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    "compile": False, # use PyTorch 2.0 to compile the model to be faster
}

# model save-load, wandb, loss-progress, distributed training etc.
logistics_config = {
    # wandb
    "wandb_log": True,
    "wandb_project": "tstok",
    "wandb_run_name": None,

    # I/O
    "out_dir": None, # filled by the script
    "eval_interval": 100,
    "log_interval": 25,
    "eval_iters": 20,
    "eval_only": False ,            # if True, script exits right after the first eval
    "always_save_checkpoint": False, # if True, always save a checkpoint after each eval

}

all_config = {
    "data": data_config,
    "model": model_config,
    "training": training_config,
    "io": logistics_config,
}
