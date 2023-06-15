
import torch
import numpy as np
from model import GPT
from generic import Config
from tokenizer import Tokenizer


def load_stuff(path, device='cpu'):
    checkpoint = torch.load(path, map_location=torch.device(device))
    cfg = Config(config=checkpoint['config'])
    tokenizer = Tokenizer(cfg.data)
    model = GPT(cfg.model)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, tokenizer

def gen_forecast(series, model, tokenizer, horizon, device, temperature=0.99, top_k=100):
    model.eval()
    input_ids, params = tokenizer.encode(series)
    with torch.no_grad():
        out = model.generate(torch.tensor(input_ids).reshape(1, -1).to(device),
                                    max_new_tokens=horizon,
                                    temperature=temperature, top_k=top_k)
    out = out.detach().cpu().numpy().ravel()
    y = tokenizer.decode(out, params)
    return y


'''
checkpoint_path = '<path to model checkpoint>'
model, tokenizer = load_checkpoint(checkpoint_path)

series = np.sin(np.linspace(-10, 10, 200)) + (np.random.rand(200) * 0.1)
fc = gen_forecast(series, model, tokenizer, 40)

plt.plot(fc, '--')
plt.plot(series)

'''
