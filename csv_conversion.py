import glob
import pandas as pd
from alias import alias
path = "./csv/*.csv"
f = open("discord-export.txt", "w+")
for fname in glob.glob(path):
    df = pd.read_csv(fname)
    for current_row_index in range(0, df.shape[0]):
        #f.write()
        author = df["Author"][current_row_index]
        content = df["Content"][current_row_index]
        f.write(str(content)+" - ")
        f.write(alias[author] + "#\n")
    print(fname)
f.close()