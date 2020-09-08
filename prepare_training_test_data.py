import pandas as pd

df = pd.read_csv("./data.csv", header=None)

df = df[0:10]
df.to_csv('training.csv', index=False, header=False)
