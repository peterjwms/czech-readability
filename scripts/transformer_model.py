from pathlib import Path
import re

from datasets import ClassLabel
from datasets import load_dataset
from datasets import Dataset
from datasets import Features
from datasets import Value
import evaluate
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import pipeline
from transformers import Trainer
from transformers import TrainingArguments
from grouped_fold_models import create_folds


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# DATA_FILE = "datasets/balanced_dataset.csv"
DATA_FILE = "datasets/full_dataset.csv" # this dataset has the full text of every doc plus all the features already extracted


def get_book_title(filename):
    parts = filename.split("_")
    id_len = len(parts[0])
    if id_len == 4:
        return parts[3]
    elif id_len == 5:
        return "CTDC"
    else:
        raise ValueError(f"{filename} has an id with the wrong number of characters.")


NUM_CLASSES = 3
# columns = {
#     "label": ClassLabel(num_classes=NUM_CLASSES, names=[0, 1, 2]),
#     "name": Value("string"),
#     "text": Value("string"),
# }

checkpoint = "xlm-roberta-base"

# features = Features(columns)
pandas_ds = pd.read_csv(Path(DATA_FILE), index_col="name")
ds = load_dataset("csv", data_files=DATA_FILE)
print(ds)
book_titles = list(set([get_book_title(name) for name in ds["train"]["name"]]))
print(f"{book_titles=}")

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_preds):
    metric = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# ds = ds["train"].train_test_split(test_size=0.2, stratify_by_column="label")
models = {}
algo_results = {}
results = {}
progress = 0

# create the groups (for the folds) and hold out a validation set
# use the same groups of data that I already have in balanced_feats_rand_groups.csv - implemented in full_dataset.csv which has group labels already now
test_fold, training_folds, group_labels = create_folds(pandas_ds)


# for val_title, test_title in zip(book_titles, book_titles[1:]): # for each group (cross-validation)

for i, fold in enumerate(training_folds):
    print("=" * 79)
    print(f"{fold=}")
    print(f"{test_fold=}")

    # TODO: adjust how we get data here
    val = ds["train"].filter(lambda x: x['group'] in fold)
    # print(val)
    test = ds["train"].filter(lambda x: x['group'] in test_fold)
    # print(test)
    train = ds["train"].filter(lambda x: x not in val and x not in test)
    # print(train)
    

    tokenized_test = test.map(tokenize_function, batched=True)
    tokenized_test.set_format(
        "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
    )
    tokenized_val = val.map(tokenize_function, batched=True)
    tokenized_val.set_format(
        "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
    )
    train = train.map(tokenize_function, batched=True)
    train.set_format(
        "pt", columns=["input_ids", "attention_mask"], output_all_columns=True
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(output_dir="models")

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=NUM_CLASSES
    ).to(device)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    algo_results[fold] = trainer.train() # this object should have the results already - holds a dict of the metrics - just the cross-validation metrics?
    # p = pipeline(model=model, task='text-classification', tokenizer=tokenizer, device=device)
    trainer.evaluate()
    # TODO: can just skip this part - this is testing on our true test set every time and giving predictions
    predictions = trainer.predict(tokenized_test)

    metric = evaluate.combine(
        [
            evaluate.load("accuracy", average="weighted"),
            evaluate.load("precision", average="weighted"),
            evaluate.load("recall", average="weighted"),
            evaluate.load("f1", average="weighted"),
        ]
    )
    results[f"{test_fold}{i=}"] = metric.compute(
        predictions=predictions.label_ids, references=test["label"]
    )

    progress += 1
    print(f"PROGRESS: {progress / len(book_titles)}")

for test_book, train_result in results.items():
    print(test_book, train_result.metrics, dir(train_result))
