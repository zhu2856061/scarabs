# -*- coding: utf-8 -*-
# @Time   : 2024/08/28 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os

from transformers import HfArgumentParser

from scarabs.args_factory import (
    DataArguments,
    ModelArguments,
    TaskArguments,
    TrainArguments,
)
from scarabs.mora.models.ctr_with_autint import CtrWithAutoInt, CtrWithAutoIntConfig
from scarabs.task_factory import TaskFactoryWithTabularCtr

parser = HfArgumentParser(
    (TaskArguments, DataArguments, ModelArguments, TrainArguments)  # type: ignore
)
task_args, data_args, model_args, training_args = parser.parse_json_file(
    "arguments.json"
)

# # Generate a tabular mapping table named 'config.json'
config = CtrWithAutoIntConfig.from_pretrained("config.json")
task = TaskFactoryWithTabularCtr(task_args, data_args, None, None, config)
task.create_feature2transformer_and_config()

# # Train
config = CtrWithAutoIntConfig.from_pretrained(
    os.path.join(
        task_args.task_name_or_path, data_args.dataset_cache, "meta/config.json"
    )
)
task = TaskFactoryWithTabularCtr(
    task_args, data_args, model_args, training_args, config
)
task.train(model=CtrWithAutoInt(config))


# # Predict
# task = TaskFactoryWithTabularCtr()
# model_path = "./encode/model"
# task.inference_with_load_model(model_path, CtrWithFM)

# import pandas as pd
# from sklearn.metrics import roc_auc_score

# preds = []
# label = []
# ds = pd.read_csv("../../data/movielens/valid/valid.csv")
# for line in ds.to_dict("records"):
#     label.append(line["label"])
#     res = task.inference(X=line)
#     preds.append(res["logits"][0].item())
# print(roc_auc_score(label, preds))


# torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py
