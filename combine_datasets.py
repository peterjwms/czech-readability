from pathlib import Path
import pandas as pd


with open(Path('datasets/balanced_dataset.csv'), encoding='UTF8') as ds_file:
    ds_text = pd.read_csv(ds_file, keep_default_na=False, na_values=['_'], sep=',', index_col='name')

with open(Path('datasets/balanced_feats_rand_groups.csv'), encoding='UTF8') as ds_file:
    ds_feats = pd.read_csv(ds_file, keep_default_na=False, na_values=['_'], sep=',', index_col='name')

full_ds = ds_feats.join(ds_text["text"])
print(full_ds.head(10))

full_ds.to_csv(Path("datasets/full_dataset.csv"))

