# -*- coding: utf-8 -*-
# @Time   : 2024/08/28 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append("../../..")

import torch
from torchinfo import summary
from transformers import HfArgumentParser

from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.mora.models.recall_with_dssm import RecallWithDSSM, RecallWithDSSMConfig
from scarabs.task_factory import TaskFactoryWithTabularRecall2

task = TaskFactoryWithTabularRecall2()

# Generate a tabular mapping table named 'config. json'
# parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
# model_args, data_args, training_args = parser.parse_json_file("arguments.json")
# config = RecallWithDSSMConfig.from_pretrained("config.json")
# task.create_feature2transformer_and_config(data_args, training_args, config)

# Train
parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
model_args, data_args, training_args = parser.parse_json_file("arguments.json")
config = RecallWithDSSMConfig.from_pretrained(
    os.path.join(data_args.dataset_cache, "meta")
)
model = RecallWithDSSM(config)
# inputs = {}
# for name, fe in config.features.items():
#     inputs[name] = torch.randint(0, 1, (2, fe["length"]))
# summary(
#     model,
#     input_data=inputs,
#     depth=5,
# )
task.train(model_args, data_args, training_args, model=model, config=config)


# torchrun --standalone --nnodes=1 --nproc_per_node=1 main.py
