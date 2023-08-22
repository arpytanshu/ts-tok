
#%%

import os
import torch
import numpy as np
import pandas as pd
from time import time, strftime
import matplotlib.pyplot as plt
import torch.nn.functional as F

from m4_exp.configurations import all_config
from m4_exp.data import get_dataloaders, data_name_mapping, get_stacked_series
from m4_exp.utils import get_hf_model, validation, sMAPE, MASE
from tstok.tokenizer import Tokenizer
from tstok.generic import Config, progress_bar



# script args
M4_DATASET_PATH = "/shared/datasets/m4_dataset"
DATANAME = "hourly"
CHECKPOINT_PATH = "/shared/CO/arpytanshu_/ts-tok/checkpoints/dev7/chkpt.pt"




print(f"{CHECKPOINT_PATH=} Attempting load...")

chkpt = torch.load(CHECKPOINT_PATH)

# load config
cfg = Config(config=all_config)
cfg.load_state_dict(chkpt['config'])
# init tokenizer using config
tokenizer = Tokenizer(cfg.data)
# load model
model = get_hf_model(cfg, tokenizer.vocab_size)
model.load_state_dict(chkpt['model'])
# init dataloader using tokenizer
dataloaders = get_dataloaders(M4_DATASET_PATH,
                              DATANAME,
                              tokenizer,
                              cfg,
                              validation=False)
te_dl = dataloaders['test']

best_val_loss = chkpt['best_val_loss']
print(f"Loading Checkpoint with {best_val_loss=} is successful.")



#%%

plot=True
plot_save_dir = "/shared/CO/arpytanshu_/ts-tok/explorations/m4_plots/generations_dev7/"

test_sequences = get_stacked_series(M4_DATASET_PATH, DATANAME)

MSL = cfg.data.max_seq_len
data_period = data_name_mapping[DATANAME]['m']

metric_MASE = []
metric_sMAPE = []
elapsed = []

num_series  = len(test_sequences['all_series'])   

for ix in range(num_series):
    tick = time()
    series = test_sequences['all_series'][ix]
    sample_len = test_sequences['sample_lengths'][ix]
    horizon = test_sequences['test_lengths'][ix]

    context = series[-horizon-MSL:-horizon]
    for test_ix in range(horizon):
        input_seq = context[-MSL:]
        input_ids, params = tokenizer.encode(input_seq)
        input_ids = torch.tensor(input_ids).to(torch.long).view(1, -1).to(model.device)

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=False, output_attentions=False, use_cache=False)
            probs = torch.softmax(out.logits[:, -1, :], dim=1).cpu()

        out_id = torch.multinomial(probs, num_samples = 1)
        generation = tokenizer.decode(out_id.detach().cpu().numpy(), params).squeeze()
        context = np.hstack([context, generation])

    smape  = sMAPE(y=series[-horizon:],
                   y_hat=context[-horizon:],
                   horizon=horizon)
    mase = MASE(context=series[:-horizon],
                y=series[-horizon:],
                y_hat=context[-horizon:],
                horizon=horizon,
                m=data_period)
    metric_MASE.append(mase)
    metric_sMAPE.append(smape)
    elapsed.append(time()  - tick)

    if plot:
        plt.figure(figsize=(20, 4))
        plt.plot(range(MSL), series[-horizon-MSL:-horizon], c='b', label='context_series')
        plt.plot(range(MSL, MSL+horizon), series[sample_len:], '--', c='b', label='test_series')
        plt.plot(range(MSL, MSL+horizon), context[-horizon:], c='r', alpha=0.6, label='generation')
        plt.title(str(ix))
        plt.legend()
        plt.savefig(os.path.join(plot_save_dir, str(ix)+'.png'))
        plt.close()
    progress_bar(ix, num_series, 50)
    break;

print()
print(f"Average MASE :: {sum(metric_MASE) / len(metric_MASE)}")
print(f"Average sMAPE :: {sum(metric_sMAPE) / len(metric_sMAPE)}")
print(f"Average elapsed :: {sum(elapsed) / len(elapsed)}")


