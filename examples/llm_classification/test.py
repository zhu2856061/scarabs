from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files=["../data/cornell_movie_review/train/train.parquet"],
    split="train",
)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("../data/albert-base-v2")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "../data/albert-base-v2", num_labels=2
)


import evaluate
import numpy as np

metric = evaluate.load("../../scarabs/metrics/accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


import sys

from transformers import HfArgumentParser, Trainer, TrainingArguments

sys.path.append("../..")
from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.train_factory import TrainerFactoryWithLLMClassification

parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
model_args, data_args, training_args = parser.parse_json_file("arguments.json")

# training_args = TrainingArguments(
#     output_dir="test_trainer", eval_strategy="epoch", per_device_train_batch_size=2
# )
trainer = TrainerFactoryWithLLMClassification(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    compute_metrics=compute_metrics,
)
trainer.train()
