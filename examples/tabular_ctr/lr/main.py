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

# from scarabs.mora.models.ctr_with_dnn import CtrWithDNN, CtrWithDNNConfig
from scarabs.mora.models.ctr_with_lr import CtrWithLR, CtrWithLRConfig
from scarabs.task_factory import TaskFactoryWithTabularCtr

task = TaskFactoryWithTabularCtr()

# # Generate a tabular mapping table named 'config. json'
# parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
# model_args, data_args, training_args = parser.parse_json_file("arguments.json")
# config = CtrWithLRConfig.from_pretrained("config.json")
# task.create_feature2transformer_and_config(data_args, training_args, config)


# # Train
# parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
# model_args, data_args, training_args = parser.parse_json_file("arguments.json")
# config = CtrWithLRConfig.from_pretrained(os.path.join(data_args.dataset_cache, "meta"))
# model = CtrWithLR(config)
# inputs = {}
# for name, fe in config.features.items():
#     inputs[name] = torch.randint(0, 1, (2, fe["length"]))
# summary(
#     model,
#     input_data=inputs,
#     depth=5,
# )
# task.train(model_args, data_args, training_args, model=model, config=config)


# # Predict
# model_path = "./encode/model/checkpoint-33957"
# task.inference_with_load_model(model_path, CtrWithLR)

# import pandas as pd

# preds = []
# label = []
# ds = pd.read_csv("../../data/movielens/test/test.csv")
# for line in ds.to_dict("records"):
#     label.append(line["label"])
#     res = task.inference(X=line)
#     preds.append(res["logits"][0].item())

# from sklearn.metrics import roc_auc_score

# print(roc_auc_score(label, preds))

# torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py
