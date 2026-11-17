"""Extrait ~10 % des clients (sur tout le mois), lecture par blocs : faible empreinte mémoire."""
import pandas as pd

SRC = "data/raw/2019-Oct.csv"
DST = "data/raw/oct_sample10.csv"

first = True
total = 0
for chunk in pd.read_csv(SRC, chunksize=1_000_000):
    part = chunk[chunk["user_id"] % 10 == 0]
    part.to_csv(DST, mode="w" if first else "a", header=first, index=False)
    first = False
    total += len(part)
    print(f"{total:,} lignes conservées")