from pathlib import Path
import pandas as pd

work_dir = Path("czech_readability_corpus")

dataset = []

for fname in Path("chapters").glob("*.txt"):
    with open(fname, encoding='UTF8') as f:
        text = f.read()
        if 'a1' in fname.name:
            dataset.append([0, fname.name, text])
        elif 'b1' in fname.name:
            dataset.append([1, fname.name, text])

i = 0
for fname in Path("ctdc").glob("*.txt"):
    if i < 230:
        with open(fname, encoding="UTF8") as f:
            text = f.read()
            dataset.append([2, fname.name, text])
            i += 1

dataset_df = pd.DataFrame(dataset, columns=["label", "name", "text"])

dataset_df.to_csv(Path("datasets/balanced_dataset.csv"), index=False)
dataset_df.to_json(Path("datasets/balanced_dataset.json"))