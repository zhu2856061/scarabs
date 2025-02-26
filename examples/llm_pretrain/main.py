# -*- coding: utf-8 -*-
# @Time   : 2024/08/21 10:24
# @Author : zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

import os
import sys

sys.path.append("../..")

import torch
from torchinfo import summary
from transformers import AutoConfig, HfArgumentParser
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from scarabs.args_factory import DataArguments, ModelArguments, TrainArguments
from scarabs.task_factory import TaskFactoryWithPreTrain

# Params
parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments))  # type: ignore

model_args, data_args, training_args = parser.parse_json_file("arguments.json")

# define model
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "token": model_args.token,
    "trust_remote_code": model_args.trust_remote_code,
}
config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
model = LlamaForCausalLM(config)

# check model
input_ids = torch.randint(0, 10, (2, 512))
attention_mask = torch.ones_like(input_ids)
summary(
    model,
    input_data={"input_ids": input_ids, "attention_mask": attention_mask},
    depth=5,
)

# train
task = TaskFactoryWithPreTrain()
task.train(model_args, data_args, training_args, model=model, config=config)


# # predict
# task = TaskFactoryWithPreTrain()
# task.inference_with_load_model(
#     "../data/llama-160m",
#     "../data/llama-160m",
#     LlamaForCausalLM,
# )
# res = task.inference("布偶猫", max_tokens=2000)
# print(res)

# accelerate launch --num_processes=2 main.py
# torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py arguments.json
# accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} main.py --all_arguments_of_the_script
