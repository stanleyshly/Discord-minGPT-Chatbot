import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset, DataLoader
import pickle
from mingpt.utils import inference, sample

class CharDataset(Dataset):

    def __init__(self, block_size):
        with open('dataset.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            self.stoi, self.itos, self.block_size, self.vocab_size, chars, data_size, vocab_size = pickle.load(f)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))


block_size = 128 # spatial extent of the model for its context
train_dataset = CharDataset(block_size) # one line of poem is roughly 50 characters
from mingpt.model import GPT

context = "Hello guys, what's up?"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPT.load_from_checkpoint("model.ckpt").to(device)
model.eval()


x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(device)

y = inference(model, x, 1000, train_dataset, temperature=0.9, sample=True)
print(y)


