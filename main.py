# the os module helps us access environment variables
# i.e., our API keys
import os

# these modules are for querying the Hugging Face model
import json
import requests

# the Discord Python API
import discord

from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from mingpt.utils import inference
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
import yaml

class CharDataset(Dataset):

    def __init__(self, block_size):
        with open('dataset.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
            self.stoi, self.itos, self.block_size, self.vocab_size, chars, data_size, vocab_size = pickle.load(f)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

class MyClient(discord.Client):
    def __init__(self, config):
        super().__init__()
        block_size = 128 # spatial extent of the model for its context
        self.train_dataset = CharDataset(block_size) # one line of poem is roughly 50 characters
        from mingpt.model import GPT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPT.load_from_checkpoint("model.ckpt").to(self.device)
        self.model.eval()
        self.config = config

    def query(self, message):
        """
        make request to the Local model 
        """
        x = torch.tensor([self.train_dataset.stoi[s] for s in message], dtype=torch.long)[None,...].to(self.device)

        response = inference(self.model, x, 1000, self.train_dataset, temperature=float(self.config["TEMPERATURE"]), sample=bool(self.config["WHITELIST_CHANNEL"]))
        return response

    async def on_ready(self):
        # print out information when the bot wakes up
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        """
        this function is called whenever the bot sees a message in a channel
        """
        # ignore the message if it comes from the bot itself
        if message.author.id == self.user.id:
            return

        if message.channel.id not in self.config["WHITELIST_CHANNEL"]:
            return
        # while the bot is waiting on a response from the model
        # set the its status as typing for user-friendliness
        async with message.channel.typing():
          response = self.query(message.content)
        
        # send the model's response to the Discord channel
        await message.channel.send(response)

def main():
    config = yaml.safe_load(open("./config.yml"))
    print(config)
    client = MyClient(config)
    client.run(config["DISCORD_TOKEN"])

if __name__ == '__main__':
  main()