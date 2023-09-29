from pathlib import Path

import evaluate
import numpy as np
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from datasets import ClassLabel, Features, Value, load_dataset

columns = {
    "label": ClassLabel(num_classes=2, names=[0, 1]),
    "name": Value("string"),
    "text": Value("string")
    }

checkpoint = "xlm-roberta-base"

features=Features(columns)
ds = load_dataset("csv", data_files="datasets/dataset.csv", features=features)
ds = ds["train"].train_test_split(test_size=0.2, stratify_by_column="label")

print(ds)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

print(tokenizer("This is an example sentence"))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = ds.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(output_dir="models")

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(model, training_args, 
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)

trainer.train()

