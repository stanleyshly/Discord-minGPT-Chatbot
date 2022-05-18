# Lexi Bot

Takes exports from Discord Chat Discord Chat Explorer and saves them as text files for minGPT training

csv_dictionary reads all and finds usernames, give option to write alias instead

csv_conversation takes that and writes the discord file to a txt
Then train and create config.yml file
```
DISCORD_TOKEN: TOKEN
WHITELIST_CHANNEL: [CHANNEL_ID]
SAMPLE: True
TEMPERATURE: 0.9
```
Then run and 

The Discord bot code is adopted from [here](https://github.com/RuolinZheng08/twewy-discord-chatbot/blob/main/discord_bot.py).

The minGPT bot code is from [here](https://github.com/karpathy/minGPT), and the code with the Pytorch-Lightning modification is from [here](https://github.com/williamFalcon/minGPT).
