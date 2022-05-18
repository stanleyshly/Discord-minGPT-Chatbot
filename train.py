import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset, DataLoader

import pickle

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

        self.pkl_archive = [self.stoi, self.itos, self.block_size, self.vocab_size, chars, data_size, vocab_size]
        with open('dataset.pkl', 'wb') as file:
            # A new file will be created
            pickle.dump(self.pkl_archive, file)


    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


block_size = 128 # spatial extent of the model for its context
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
text = open('discord-export.txt', 'r').read() # don't worry we won't run out of file handles
train_dataset = CharDataset(text, block_size) # one line of poem is roughly 50 characters
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=4)
from mingpt.model import GPT
model = GPT(vocab_size=train_dataset.vocab_size, 
            block_size=train_dataset.block_size,
            n_layer=12, 
            n_head=12, 
            n_embd=768)#GPT-1, 
            #learning_rate=6e-4)

from pytorch_lightning import Trainer
from mingpt.lr_decay import LearningRateDecayCallback
from pytorch_lightning.callbacks import ModelCheckpoint

lr_decay = LearningRateDecayCallback(
    learning_rate=6e-4,
    warmup_tokens=512 * 20,
    final_tokens=2 * len(train_dataset) * block_size
)
trainer = Trainer(gpus=1, 
                  precision=16,
                  max_epochs=500, 
                  callbacks=[lr_decay, ModelCheckpoint(every_n_train_steps=100)],
                  gradient_clip_val=1.0,
                  accumulate_grad_batches=32)#,
                  #resume_from_checkpoint = "./model.ckpt")

# every_n_training_steps is different cause gradeiaccumulation

#lr_finder = trainer.tuner.lr_find(model, train_loader)
#new_lr = lr_finder.suggestion()
#print(new_lr)
trainer.fit(model, train_loader)
