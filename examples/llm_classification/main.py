# -*- coding: utf-8 -*-
# @Time   : 2024/08/28 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import sys

sys.path.append("../..")

import torch
from torchinfo import summary
from transformers import AlbertConfig, AlbertForSequenceClassification, HfArgumentParser

from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.task_factory import TaskFactoryWithLLMClassification

parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
model_args, data_args, training_args = parser.parse_json_file("arguments.json")

# define model
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "token": model_args.token,
    "trust_remote_code": model_args.trust_remote_code,
    "num_labels": data_args.num_labels,
}
config = AlbertConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
model = AlbertForSequenceClassification(config)  # type: ignore

# check model
input_ids = torch.randint(0, 10, (2, 512))
attention_mask = torch.ones_like(input_ids)
summary(
    model,
    input_data={"input_ids": input_ids, "attention_mask": attention_mask},
    depth=5,
)

# train
task = TaskFactoryWithLLMClassification()
task.train(model_args, data_args, training_args, model=model, config=config)


# # predict
# task = TaskFactoryWithLLMClassification()
# task.inference_with_load_model(
#     "../data/albert-base-v2",
#     "encode/model/checkpoint-200",
#     AlbertForSequenceClassification,
# )
# res = task.inference("hello world")
# print(res)
