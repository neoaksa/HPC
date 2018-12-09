import pandas as pd

a = pd.read_csv("part-00000-a7ef622e-4ce8-460e-8872-f02cf7717499-c000.csv", header=0)
b = pd.read_csv("part-00000-d319db39-9095-429b-9c02-6e1b168f8bf2-c000.csv", header=0)
b = b.dropna(axis=1)
merged = a.append(b)
merged.to_csv("output.csv", index=False)