# -*- coding: utf-8 -*-
# @Time   : 2024/08/28 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os

from transformers import HfArgumentParser

from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.mora.models.classification_with_dnn import (
    ClassificationWithDNN,
    ClassificationWithDNNConfig,
)
from scarabs.task_factory import TaskFactoryWithTabularClassification

task = TaskFactoryWithTabularClassification()


# Generate a tabular mapping table named 'config. json'
# Params
parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
model_args, data_args, training_args = parser.parse_json_file("arguments.json")
config = ClassificationWithDNNConfig.from_pretrained("config.json")
task.create_feature2transformer_and_config(data_args, training_args, config)


# Train
parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore
model_args, data_args, training_args = parser.parse_json_file("arguments.json")
config = ClassificationWithDNNConfig.from_pretrained(
    os.path.join(data_args.dataset_cache, "meta")
)
model = ClassificationWithDNN(config)

task.train(model_args, data_args, training_args, model=model, config=config)


# # Predict
# model_path = "./encode/model/checkpoint-800"
# task.inference_with_load_model(model_path, ClassificationWithDNN)
# res = task.inference(X={})
# print(res)
